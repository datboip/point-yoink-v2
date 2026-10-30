# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finding the scanner on the local network for its motion stream, and remembering where it was.

The scanner streams motion on TCP port 9999 over Wi-Fi. Its address is remembered in the app's config.json as
scanner_ip (the Wi-Fi transfer saves it too when the scanner searches for this PC). When the remembered address is
gone, the PC's neighbors and then the rest of its /24 network are tried, and only an address that really sends a
motion packet counts."""
import ipaddress
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app2.scan.telemetry import decode_packet

PORT = 9999


def _config():
    return Path(os.environ.get('POINTYOINK_CONFIG') or Path.home() / '.config/pointyoink/config.json')


def remember(ip):
    """Save the scanner's address as scanner_ip, keeping every other setting."""
    try:
        ip = str(ipaddress.ip_address(str(ip).strip()))
    except ValueError:
        return False
    path = _config()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if data.get('scanner_ip') == ip:
        return True
    data['scanner_ip'] = ip
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.scanner_ip.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
    except OSError:
        return False
    return True


def own_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except OSError:
        return ''
    finally:
        s.close()


def neighbors(table='/proc/net/arp'):
    """Addresses this PC has talked to lately on the network (complete entries only)."""
    found = []
    try:
        lines = Path(table).read_text().splitlines()[1:]
    except OSError:
        return found
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2] != '0x0':
            found.append(parts[0])
    return found


def subnet_hosts(ip):
    try:
        net = ipaddress.ip_network(ip + '/24', strict=False)
    except ValueError:
        return []
    return [str(h) for h in net.hosts() if str(h) != ip]


def speaks_motion(ip, port=PORT, connect_timeout=.4, read_timeout=1.2):
    """True when something at ip:port sends a motion packet that decodes."""
    try:
        sock = socket.create_connection((ip, port), timeout=connect_timeout)
    except OSError:
        return False
    try:
        sock.settimeout(read_timeout)
        buffer = b''
        while len(buffer) < 60:
            chunk = sock.recv(4096)
            if not chunk:
                return False
            buffer += chunk
        decode_packet(buffer[:60])
        return True
    except (OSError, ValueError):
        return False
    finally:
        sock.close()


def candidates(saved='', own='', arp=None):
    """Where to look, most likely first: the remembered address, the neighbors, then the rest of the network."""
    own = own or own_ip()
    order = []
    for ip in [saved] + list(neighbors() if arp is None else arp) + subnet_hosts(own):
        if ip and ip != own and ip not in order:
            order.append(ip)
    return order


def find(saved='', stop=None, port=PORT, workers=48, own='', arp=None):
    """The scanner's address, or '' when nothing on the network answers like it. `stop` is an optional Event."""
    from concurrent.futures import as_completed
    if saved and speaks_motion(saved, port):
        return saved
    order = [ip for ip in candidates('', own, arp) if ip != saved]
    if not order:
        return ''
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        jobs = {pool.submit(lambda ip=ip: not (stop is not None and stop.is_set()) and speaks_motion(ip, port)): ip for ip in order}
        for done in as_completed(jobs):
            if done.result():
                return jobs[done]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return ''
