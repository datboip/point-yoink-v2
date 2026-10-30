# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only, opt-in device motion stream. Raw fields require calibration for pose hints.

The locally observed packet has an opaque eight-byte header and 13 float32s.
Axes, units, and synchronization are not a published protocol contract.
"""
from dataclasses import dataclass
from collections import deque
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import struct
import threading
import time


def configured_ip():
    """Read the existing setting without importing the Tk app or writing config."""
    path = Path(os.environ.get('POINTYOINK_CONFIG') or Path.home()/'.config/pointyoink/config.json')
    try:
        value = json.loads(path.read_text()).get('scanner_ip', '')
        return str(ipaddress.ip_address(value))
    except (OSError, ValueError, TypeError, AttributeError):
        return ''


def quaternion_multiply(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)


def euler_degrees(q):
    x, y, z, w = q
    heading = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    return tuple(math.degrees(v) for v in (heading, pitch, roll))


def configured_zero():
    """The saved level pose (a unit quaternion x, y, z, w), or None."""
    path = Path(os.environ.get('POINTYOINK_CONFIG') or Path.home()/'.config/pointyoink/config.json')
    try:
        q = tuple(float(v) for v in json.loads(path.read_text()).get('motion_zero'))
        if len(q) == 4 and all(math.isfinite(v) for v in q) and abs(math.sqrt(sum(v*v for v in q)) - 1) < 1e-3: return q
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def _config_path():
    return Path(os.environ.get('POINTYOINK_CONFIG') or Path.home()/'.config/pointyoink/config.json')


def _save_key(key, value):
    """Set (or with None, remove) one top-level setting, keeping every other one."""
    path = _config_path()
    try: data = json.loads(path.read_text())
    except (OSError, ValueError): data = {}
    if not isinstance(data, dict): data = {}
    if value is None: data.pop(key, None)
    else: data[key] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.motion.tmp'); tmp.write_text(json.dumps(data, indent=2)); tmp.replace(path)
    except OSError:
        return False
    return True


def save_zero(q):
    """Remember the level pose (None forgets it), keeping every other setting."""
    return _save_key('motion_zero', None if q is None else [float(v) for v in q])


def _unit(v):
    n = math.sqrt(sum(a*a for a in v))
    return tuple(a/n for a in v) if n > 1e-9 else None


def _dot(a, b): return sum(x*y for x, y in zip(a, b))


def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def axes_from(flat_acceleration, down_acceleration):
    """The scanner's own up, forward and right in the sensor's axes. Flat on its base the sensor reads gravity along
    the scanner's up; pointed straight down it reads gravity against the scanner's forward. None when the two poses
    do not look like those (the readings are missing, or the poses are not a quarter turn apart)."""
    up = _unit(flat_acceleration); down = _unit(down_acceleration)
    if up is None or down is None: return None
    forward = tuple(-a for a in down)
    along = _dot(up, forward)
    if abs(along) > .4: return None
    forward = _unit(tuple(f - along*u for f, u in zip(forward, up)))
    return dict(up=up, forward=forward, right=_cross(forward, up))


def body_angles(axes, acceleration, quaternion, zero):
    """Heading, pitch, roll in degrees. Pitch is how far the scanner's forward points above level (negative: down) and
    roll how far its right side is down, both from gravity; heading is the turn about up since the level pose."""
    a = _unit(acceleration)
    if a is None or axes is None: return None
    pitch = math.degrees(math.asin(max(-1., min(1., _dot(a, axes['forward'])))))
    roll = math.degrees(math.atan2(-_dot(a, axes['right']), _dot(a, axes['up'])))
    x, y, z, w = quaternion if zero is None else quaternion_multiply((-zero[0], -zero[1], -zero[2], zero[3]), quaternion)
    heading = math.degrees(2*math.atan2(_dot((x, y, z), axes['up']), w))
    heading = (heading + 180) % 360 - 180
    return heading, pitch, roll


def configured_axes():
    try:
        d = json.loads(_config_path().read_text()).get('motion_axes')
        axes = {k: tuple(float(v) for v in d[k]) for k in ('up', 'forward', 'right')}
        if all(len(v) == 3 and all(math.isfinite(c) for c in v) for v in axes.values()): return axes
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        pass
    return None


def save_axes(axes):
    return _save_key('motion_axes', None if axes is None else {k: [float(c) for c in axes[k]] for k in ('up', 'forward', 'right')})


def mean_pose(samples):
    """(quaternion, acceleration, spread in degrees) over a few samples held still: the average pose, and how much
    the pose wandered while it was held (a still scanner reads under about a degree)."""
    if not samples: raise ValueError('No samples')
    ref = samples[0].quaternion; total = [0., 0., 0., 0.]
    for s in samples:
        q = s.quaternion
        if sum(a*b for a, b in zip(q, ref)) < 0: q = tuple(-v for v in q)       # q and -q are the same turn
        total = [a+b for a, b in zip(total, q)]
    norm = math.sqrt(sum(v*v for v in total)) or 1.
    mean = tuple(v/norm for v in total)
    accel = tuple(sum(s.acceleration[k] for s in samples)/len(samples) for k in range(3))
    spread = max(max(abs(a) for a in Sample_angles_between(mean, s.quaternion)) for s in samples)
    return mean, accel, spread


def Sample_angles_between(a, b):
    x, y, z, w = a
    return euler_degrees(quaternion_multiply((-x, -y, -z, w), b))


@dataclass(frozen=True)
class Sample:
    quaternion: tuple
    gyro: tuple
    acceleration: tuple
    position: tuple
    received: float

    @property
    def angles(self):
        """Relative heading, pitch, roll in degrees; heading is not magnetic north."""
        return euler_degrees(self.quaternion)

    def angles_from(self, zero=None):
        """Heading, pitch, roll relative to a saved level pose (the raw angles when there is none)."""
        if zero is None: return self.angles
        x, y, z, w = zero
        return euler_degrees(quaternion_multiply((-x, -y, -z, w), self.quaternion))


def decode_packet(packet, received=None):
    if len(packet) != 60:
        raise ValueError('Expected a 60-byte motion packet')
    values = struct.unpack_from('<13f', packet, 8)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite motion packet')
    norm = math.sqrt(sum(v*v for v in values[:4]))
    if norm < 1e-8:
        raise ValueError('Empty orientation')
    return Sample(tuple(v/norm for v in values[:4]), values[4:7], values[7:10],
                  values[10:13], time.monotonic() if received is None else received)


class TelemetryClient:
    """One daemon reader and one latest sample, with bounded receive storage."""
    def __init__(self, stale_after=2.):
        self.stale_after = stale_after
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._socket = None
        self._sample = None
        self._state = 'Disconnected'
        self._count = 0;self._history=deque();self._recorder=None

    def set_recorder(self,recorder):
        with self._lock:self._recorder=recorder

    def ingest(self,packet,received=None):
        sample=decode_packet(packet,received)
        self._accept(sample)
        return sample

    def _accept(self,sample,stop=None):
        with self._lock:
            if stop is not None and (stop.is_set() or stop is not self._stop):return
            if self._history and sample.received<=self._history[-1].received:return
            self._sample=sample;self._count+=1;self._state='Receiving';self._history.append(sample)
            while self._history and self._history[0].received<sample.received-self.stale_after:self._history.popleft()
            recorder=self._recorder
        if recorder is not None:recorder(sample)

    def recent(self, seconds=1.):
        """The samples from the last `seconds`, oldest first."""
        with self._lock:
            if not self._history: return []
            edge = self._history[-1].received - seconds
            return [s for s in self._history if s.received >= edge]

    def snapshot_at(self,now):
        with self._lock:
            sample=next((s for s in reversed(self._history) if s.received<=now),None)
            state,count=self._state,self._count
        age=None if sample is None else now-sample.received
        if state=='Receiving':state='Waiting for data' if sample is None else ('Stale' if age>self.stale_after else state)
        return state,sample,count,age

    def connect(self, address, port=9999):
        address = str(ipaddress.ip_address(address.strip()))
        self.disconnect()
        stop = threading.Event()
        with self._lock:
            self._stop = stop
            self._state = 'Connecting'
            self._sample = None
            self._count = 0;self._history.clear()
        self._thread = threading.Thread(target=self._run, args=(address, port, stop),
                                        name='device-motion', daemon=True)
        self._thread.start()

    def disconnect(self):
        self._stop.set()
        with self._lock:
            sock, self._socket = self._socket, None
            self._state = 'Disconnected'
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    close = disconnect

    def snapshot(self, now=None):
        with self._lock:
            state, sample, count = self._state, self._sample, self._count
        age = None if sample is None else max(0., (time.monotonic() if now is None else now)-sample.received)
        if state == 'Receiving' and age is not None and age > self.stale_after:
            state = 'Stale'
        return state, sample, count, age

    def _run(self, address, port, stop):
        sock = None
        try:
            sock = socket.create_connection((address, port), timeout=2.)
            sock.settimeout(.25)
            with self._lock:
                if stop.is_set() or stop is not self._stop:
                    return
                self._socket = sock
                self._state = 'Waiting for data'
            buffer = bytearray()
            while not stop.is_set():
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer.extend(chunk)
                while len(buffer) >= 60:
                    packet = bytes(buffer[:60]); del buffer[:60]
                    try:
                        sample = decode_packet(packet)
                    except ValueError:
                        continue
                    self._accept(sample,stop)
        except OSError:
            with self._lock:
                if stop is self._stop and not stop.is_set():
                    self._state = 'Could not connect' if self._sample is None else 'Connection lost'
        finally:
            if sock:
                sock.close()
            with self._lock:
                if stop is self._stop:
                    self._socket = None
                    if self._state in ('Receiving', 'Waiting for data'):
                        self._state = 'Disconnected'
