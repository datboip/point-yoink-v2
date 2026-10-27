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
        x, y, z, w = self.quaternion
        heading = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
        roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        return tuple(math.degrees(v) for v in (heading, pitch, roll))


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
