# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where frames come from: the MIRACO in PC mode (LiveSource: plain v4l2-ctl, exactly what the Tk scanner runs) or a
saved project played back at the scanner's pace (ReplaySource). The session sees the same thing from both: Frame
objects handed to its sink on the source's own thread, the newest colour picture (JPEG bytes) on request, and gain /
fill LED settings that go to the scanner (live) or nowhere (replay). Nothing here runs on the window's thread.

Safety: with POINTYOINK_NO_DEVICE=1 in the environment, anything that would talk to the real scanner (the default
v4l2-ctl commands, the gain control, the fill LED) raises instead of running. Tests set it; the command lines can be
swapped for a stand-in streamer (argv arguments), which is how LiveSource is tested without a scanner."""
import glob, json, os, queue, subprocess, sys, threading, time
import numpy as np
from app2 import REPO
from app2.scan import fmt
from app2.scan import timing

W, H = fmt.W, fmt.H
COLOR_SYNC_S = .1   # nearest color exposure may be at most 100 ms from a depth frame
FS = W * H * 4                      # Y16 800x1200: depth u16 | IR left u8 | IR right u8
DEPTH_DEV, COLOUR_DEV = "/dev/video0", "/dev/video2"
DEPTH_ARGV = ["v4l2-ctl", "-d", DEPTH_DEV, "--set-fmt-video", "width=800,height=1200,pixelformat=Y16 ", "--stream-mmap", "--stream-to=-"]
COLOUR_ARGV = ["v4l2-ctl", "-d", COLOUR_DEV, "--set-fmt-video", "width=2000,height=1500,pixelformat=MJPG", "--stream-mmap", "--stream-to=-"]

try:
    from app2.scan.scanner_led import set_fill_led
except Exception:
    set_fill_led = None        # no LED control: the button stays off


def no_device():
    """POINTYOINK_NO_DEVICE=1: never talk to the real scanner (tests)."""
    return os.environ.get("POINTYOINK_NO_DEVICE") == "1"


def _refuse(what):
    raise RuntimeError("POINTYOINK_NO_DEVICE=1: %s refused (nothing is sent to the scanner)" % what)


def set_ir_gain(g):
    """IR sensor gain (standard UVC control on the depth camera, 1..15). The scanner keeps exposure time on its own auto
    and refuses manual exposure, so gain is the brightness knob. Measured on a dark part: gain 4 gave about 2x the depth
    of the default 1; 15 clips the IR and loses depth again."""
    if no_device(): _refuse("IR gain")
    # a timeout: a call stuck on a half-gone USB device would hold up every later gain and LED change
    subprocess.run(["v4l2-ctl", "-d", DEPTH_DEV, "--set-ctrl=gain=%d" % int(g)], capture_output=True, timeout=5)


def fill_led(on):
    if no_device(): _refuse("fill LED")
    set_fill_led(on)


class Frame:
    """One depth frame as it arrived. t: time.monotonic() at arrival (the stamp's "t"). depth: HxW u16 in 0.1 mm, not cut
    yet. irl / irr: IR left / right, HxW u8 (irr None for older replays). meta: replay only, where the frame came from (side
    folder, frame name, its saved stamp)."""
    __slots__ = ("t", "depth", "irl", "irr", "seq", "meta")

    def __init__(self, t, depth, irl, irr=None, seq=0, meta=None):
        self.t, self.depth, self.irl, self.irr, self.seq, self.meta = t, depth, irl, irr, seq, meta


def split_y16(buf, t, seq):
    """One 1,920,000-byte Y16 800x1200 frame: depth = the first 960,000 bytes, IR left the next 480,000, IR right the last."""
    d = np.frombuffer(buf, np.uint16, W * H).reshape(H, W)
    irl = np.frombuffer(buf, np.uint8, W * H, W * H * 2).reshape(H, W)
    irr = np.frombuffer(buf, np.uint8, W * H, W * H * 3).reshape(H, W)
    return Frame(t, d, irl, irr, seq)


class DeviceControl:
    """Settings for the scanner, one after another on their own thread (a v4l2-ctl call can hang on a vanished device;
    the window and the tracker never wait for one). Gain goes out only when it changed; LED calls go out in order.
    gain_fn / led_fn: what sends them (the real ones by default; a test passes recorders)."""
    def __init__(self, live, gain_fn=None, led_fn=None, present=True):
        self.live = live;self.present=bool(present);self.led_desired=False; self.q = queue.Queue(); self.gain_set = None;self.led_state="unknown";self.led_request=0
        self.gain_fn = gain_fn or set_ir_gain
        self.led_fn = led_fn or (fill_led if set_fill_led is not None else None)
        threading.Thread(target=self._run, name="device", daemon=True).start()

    def _run(self):
        while True:
            fn, arg, done = self.q.get()
            if fn is StopIteration: return
            try:
                result=fn(arg) if fn is not None else None
                if callable(done):done(result,None)
            except Exception as e:
                print("device:", e, flush=True)
                if callable(done):done(None,e)
            if done is not None and not callable(done):done.set()

    def close(self):
        """No more settings: the thread ends once the queue is done (after stop's LED off and gain 1)."""
        self.q.put((StopIteration, None, None))

    @property
    def has_led(self): return self.live and self.led_fn is not None

    def gain(self, g):
        g = int(g)
        if g == self.gain_set: return
        self.gain_set = g
        if self.live: self.q.put((self.gain_fn, g, None))

    def forget_gain(self): self.gain_set = None            # after a reconnect: send it again on the next frame

    def led(self, on):
        if self.has_led:
            self.led_desired=bool(on);self.led_request+=1;request=self.led_request;available=self.present;self.led_state='pending' if available else 'unknown'
            def finished(result,error):
                if request!=self.led_request:return
                # A queued call or a no-return setter cannot confirm the physical light.
                confirmed=result.get('on') if isinstance(result,dict) and result.get('readback') is True else None
                self.led_state=('failed' if available and self.present else 'unknown') if error else ('on' if confirmed else 'off') if isinstance(confirmed,bool) else 'unverified'
            self.q.put((self.led_fn,bool(on),finished))

    def stream_lost(self):
        self.present=False;self.led_request+=1;self.led_state='unknown'

    def stream_arrived(self):
        if self.present:return
        self.present=True;self.forget_gain()
        if self.has_led:self.led(self.led_desired)

    def leave(self):
        """Closing: LED off, then the scanner's default gain 1 (always sent, as the Tk scanner does at exit)."""
        self.led(False)
        if self.live: self.q.put((self.gain_fn, 1, None))

    def flush(self, timeout=3.0):
        ev = threading.Event(); self.q.put((None, None, ev)); return ev.wait(timeout)


class FrameSource:
    live = False
    kind = "none"
    label = ""

    def __init__(self, gain_fn=None, led_fn=None):
        self.last_t = time.monotonic(); self.dead = False; self._colour = None
        self._color_lock=threading.Lock();self._color_history=[];self._color_seq=0
        self._color_listeners=set()
        self.device = DeviceControl(self.live, gain_fn, led_fn,present=not self.live)

    @property
    def has_led(self): return self.device.has_led

    def start(self, sink): raise NotImplementedError
    def stop(self): pass
    def restart(self): pass                                # reconnect after the stream died
    def present(self): return True                         # is the device there to reconnect to
    def color_arrived(self,jpeg,t,frame_id=None):
        with self._color_lock:
            self._color_seq+=1;self._colour=jpeg
            self._color_history.append(dict(jpeg=jpeg,t=float(t),seq=self._color_seq,frame_id=frame_id))
            self._color_history=self._color_history[-16:]
            listeners=tuple(self._color_listeners)
        for wake in listeners: wake()

    def subscribe_color(self, wake):
        with self._color_lock: self._color_listeners.add(wake)
        def unsubscribe():
            with self._color_lock: self._color_listeners.discard(wake)
        return unsubscribe

    def color_sample(self,t,max_offset=COLOR_SYNC_S):
        with self._color_lock:
            if not self._color_history:return None
            frame=min(self._color_history,key=lambda x:abs(x['t']-t))
            return dict(frame) if abs(frame['t']-t)<=max_offset else None

    def latest_color(self):
        with self._color_lock:return dict(self._color_history[-1]) if self._color_history else None

    def colour(self): return self._colour                  # newest colour frame, JPEG bytes (or None)
    def ack(self): pass                                    # the session finished a frame (lockstep replay waits for it)
    def gone_text(self): return ""                         # the banner when present() is False
    def join(self, timeout=2.0): pass                     # after stop(): wait (briefly) for the source's own threads


class LiveSource(FrameSource):
    """The MIRACO in PC mode: /dev/video0 Y16 800x1200 (depth, IR left, IR right, about 3 fps) and /dev/video2 MJPG
    2000x1500 colour, both through v4l2-ctl. The streams can die or stall under us (the scanner dropped off USB mid-side,
    seen live): gen tells a pump whether it still belongs to the current connection.
    depth_argv / colour_argv: the stream commands (default: the Tk scanner's, verbatim); a test passes a streamer."""
    live = True
    kind = "live"
    label = "Scanner in PC mode"

    def __init__(self, depth_argv=None, colour_argv=None, gain_fn=None, led_fn=None, depth_dev=DEPTH_DEV):
        super().__init__(gain_fn, led_fn)
        self.default_argv = depth_argv is None and colour_argv is None
        self.depth_argv = list(depth_argv or DEPTH_ARGV); self.colour_argv = list(colour_argv or COLOUR_ARGV)
        self.depth_dev = depth_dev
        self.proc = self.rgb = None; self.gen = 0; self.running = False; self.sink = None; self.seq = 0; self.pumps = []

    def present(self): return os.path.exists(self.depth_dev)

    def gone_text(self):
        return "Scanner disconnected. Connect the USB cable and choose PC mode on the scanner. We will reconnect automatically."

    def _pump(self, p, gen):               # never drop a raw frame; the session decides what to keep
        out = p.stdout
        while self.running and self.gen == gen:
            fr = bytearray()
            while len(fr) < FS:
                c = out.read(FS - len(fr))
                if not c:
                    if self.gen == gen: self.dead = True
                    return
                fr += c
            self.last_t = time.monotonic(); self.seq += 1
            self.device.stream_arrived()
            self.sink(split_y16(bytes(fr), self.last_t, self.seq))

    def _pump_rgb(self, p, gen):           # colour is only for looking at: keep the newest frame, drop the rest
        out = p.stdout; buf = b""
        while self.running and self.gen == gen:
            # A pipe read(size) waits for the whole block, even when a complete
            # JPEG is already available. read1 consumes the available packet.
            chunk = getattr(out, 'read1', out.read)(262144)
            if not chunk: return
            buf += chunk
            while True:
                a0 = buf.find(b"\xff\xd8")
                if a0 < 0: buf = b""; break
                b0 = buf.find(b"\xff\xd9", a0 + 2)
                if b0 < 0: buf = buf[a0:]; break
                t_in = time.monotonic(); jpeg = buf[a0:b0 + 2]
                timing.mark('color_in'); timing.save_color(jpeg, t_in)
                self.color_arrived(jpeg,t_in); buf = buf[b0 + 2:]

    @staticmethod
    def _reap(p):                          # a v4l2-ctl stuck on a vanished device may ignore terminate
        try: p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired: print("v4l2-ctl %d did not exit" % p.pid, flush=True)

    def _stop_streams(self, wait=False):
        """End both streams. wait: until both processes are gone (on the worker: closing, and before a reconnect opens
        the device again, so the new stream never finds the old one still holding it)."""
        ps = [p for p in (self.proc, self.rgb) if p is not None and p.poll() is None]
        for p in ps: p.terminate()
        if wait:
            for p in ps: self._reap(p)
        else:
            for p in ps: threading.Thread(target=self._reap, args=(p,), daemon=True).start()

    def start(self, sink):
        self.sink = sink; self.running = True; self.restart()
        self.device.led(False)                         # the scanner can't report the LED state back: start from off

    def restart(self):
        if self.default_argv and no_device(): _refuse("streaming from %s" % self.depth_dev)
        self.device.stream_lost()
        self._stop_streams(wait=True); self.gen += 1; gen = self.gen
        self.proc = subprocess.Popen(self.depth_argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.rgb = subprocess.Popen(self.colour_argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.last_t = time.monotonic(); self.dead = False; self.device.forget_gain()
        self.pumps = [threading.Thread(target=self._pump, args=(self.proc, gen), name="depth-pump", daemon=True),
                      threading.Thread(target=self._pump_rgb, args=(self.rgb, gen), name="colour-pump", daemon=True)]
        for t in self.pumps: t.start()

    def stop(self):
        """Leave the scanner as found: LED off, default gain, streams closed and their processes gone. Runs on the
        session's worker (up to 3 s for the settings and 5 s per stream on a hung v4l2-ctl; the window never waits)."""
        self.device.leave()
        if not self.device.flush(3.0): print("device: LED off / gain 1 not confirmed within 3 s", flush=True)
        self.device.close()
        self.running = False; self._stop_streams(wait=True)

    def join(self, timeout=2.0):
        for t in self.pumps: t.join(timeout)


class ReplaySource(FrameSource):
    """Plays the saved sides of a project as if the scanner sent them: data/<side>/cache/*.dph + *.irl in recorded order
    (pointyoink_track.json's stamps, else by pass and frame number), about 3 fps, colour from the side's thumb_rgb.jpg.
    Never writes anything.
    sides: which sides, as 1-based numbers or folder names (default: all saved sides).
    lockstep=True: the next frame goes out when the session has finished the last one (tests, as fast as the tracker).
    loop: start over at the end. limit: at most this many frames. stall=(n, secs): stop sending for secs after the n-th
    frame, to try the watchdog and reconnect. on_before(meta): called on the player's thread just before a frame goes
    out (a test presses buttons there). skip: leave out the first frames."""
    kind = "replay"

    def __init__(self, project, sides=None, fps=3.0, lockstep=False, loop=False, limit=None, stall=None, skip=0, on_before=None):
        super().__init__(); self.on_before = on_before
        self.project = project; self.fps = fps; self.lockstep = lockstep; self.loop = loop; self.stall = stall
        self.label = "Replay: %s, %s" % (os.path.basename(os.path.normpath(project)),
                                         "in lockstep" if lockstep else "%g fps" % fps)
        self.folders = self.pick_sides(project, sides); self.skip, self.limit = skip, limit; self._plan = None
        self.credit = threading.Semaphore(1); self.running = False; self.done = threading.Event(); self.sent = 0
        self.reconnects = 0

    @staticmethod
    def pick_sides(project, sides):
        """The side folders to play: sides as 1-based numbers or folder names (None: every saved side)."""
        folders = fmt.saved_sides(project)
        if sides is None: return folders
        out = []
        for i in sides:
            if isinstance(i, int):
                if not 1 <= i <= len(folders): raise ValueError("no side %d in %s (it has %d)" % (i, project, len(folders)))
                out.append(folders[i - 1])
            else:
                f = os.path.join(project, "data", i)
                if f not in folders: raise ValueError("no saved side %s in %s" % (i, project))
                out.append(f)
        return out

    @property
    def plan(self):
        """[(side index, folder, frame name, saved stamp or None)] in playing order. Made on first use: the app first
        uses it on the player thread (reading every side's stamps must not hold up the window); a test can read or trim
        it before start()."""
        if self._plan is None:
            plan = []
            for si, f in enumerate(self.folders):
                cache = os.path.join(f, "cache")
                try:
                    with open(os.path.join(cache, fmt.TRACK_JSON)) as fh: stamps = json.load(fh).get("stamps") or []
                except Exception: stamps = []
                if stamps: items = [(s["frame"], s) for s in stamps]
                else: items = [(os.path.basename(p)[:-4], None) for p in sorted(glob.glob(os.path.join(cache, "frame_*.dph")))]
                plan += [(si, f, n, s) for n, s in items if os.path.exists(os.path.join(cache, n + ".dph"))]
            plan = plan[self.skip:]
            self._plan = plan[:self.limit] if self.limit else plan
        return self._plan

    @plan.setter
    def plan(self, v): self._plan = v

    def start(self, sink):
        self.sink = sink; self.running = True
        self.thread = threading.Thread(target=self._run, name="replay", daemon=True); self.thread.start()

    def join(self, timeout=2.0):
        t = getattr(self, "thread", None)
        if t is not None: t.join(timeout)

    def stop(self): self.running = False; self.credit.release(); self.device.close()

    def ack(self):
        self.last_t = time.monotonic()                     # the session is keeping up: not a dead stream
        if self.lockstep: self.credit.release()

    def restart(self): self.reconnects += 1; self.last_t = time.monotonic(); self.dead = False

    def present(self): return not self.done.is_set()      # played to the end: gone, nothing to reconnect to

    def gone_text(self): return "Replay finished: no more frames."

    def _run(self):
        t_next = time.monotonic(); last_side = None
        while self.running:
            for k, (si, folder, name, stamp) in enumerate(self.plan):
                if not self.running: break
                if self.lockstep: self.credit.acquire()
                elif self.fps > 0:
                    t_next += 1.0 / self.fps; time.sleep(max(0.0, t_next - time.monotonic()))
                if not self.running: break
                if self.stall and self.sent == self.stall[0]:
                    time.sleep(self.stall[1]); t_next = time.monotonic()
                if si != last_side:
                    last_side = si; p = os.path.join(folder, "thumb_rgb.jpg")
                    if os.path.exists(p):
                        with open(p, "rb") as fh: self._colour = fh.read()
                    else: self._colour = None
                cache = os.path.join(folder, "cache")
                d = np.fromfile(os.path.join(cache, name + ".dph"), np.uint16)
                irl = np.fromfile(os.path.join(cache, name + ".irl"), np.uint8)
                if d.size != W * H or irl.size != W * H: continue
                nxt = self.plan[k + 1] if k + 1 < len(self.plan) else None
                prv = self.plan[k - 1] if k > 0 else None
                meta = dict(side=si, folder=folder, frame=name, stamp=stamp, k=k,
                            first_of_side=prv is None or prv[0] != si,
                            first_of_pass=prv is None or prv[0] != si or prv[2][:9] != name[:9],
                            last_of_side=nxt is None or nxt[0] != si,
                            last_of_pass=nxt is None or nxt[0] != si or nxt[2][:9] != name[:9])
                if self.on_before is not None: self.on_before(meta)
                self.last_t = time.monotonic(); self.sent += 1
                image_path=os.path.join(cache,name+".img")
                if os.path.isfile(image_path):
                    with open(image_path,"rb") as f:self.color_arrived(f.read(),self.last_t,frame_id=name)
                right_path=os.path.join(cache,name+".irr")
                irr=np.fromfile(right_path,np.uint8) if os.path.isfile(right_path) else None
                irr=irr.reshape(H,W) if irr is not None and irr.size==W*H else None
                self.sink(Frame(self.last_t, d.reshape(H, W), irl.reshape(H, W), irr, self.sent, meta))
            if not self.loop: break
        if self.lockstep and self.running: self.credit.acquire()       # the last frame is handled too
        self.done.set()
