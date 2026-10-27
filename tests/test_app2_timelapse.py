"""Export scan timelapse (app2/scan/timelapse.py): a tiny synthetic side (a ball on a platter, the camera going around
it) is played back offscreen into MP4 and GIF clips. The model grows frame by frame, the mark sits bottom-left, the
files hold the frames and length asked for, Cancel and SIGTERM leave nothing behind. The model is fused on the CPU
(no CUDA needed); drawing uses GL on EGL when the machine has it, else the numpy splats."""
import json, os, shutil, signal, struct, subprocess, sys, tempfile, time
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink-timelapse-')) / 'config.json')
os.environ['POINTYOINK_NO_DEVICE'] = '1'
import numpy as np
import pytest
import app2  # noqa: F401 (puts the repo and scanner/ on sys.path)
from app2.scan import fmt
from app2.scan import timelapse as TL
from track import F

W, H = fmt.W, fmt.H
CAL = (560.0, 560.0, 400.0, 300.0)
RADIUS, DIST, N = 30.0, 260.0, 14
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_ffmpeg = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='no ffmpeg')


def cam_pose(k):
    """Camera k of N (OpenCV camera -> scan frame, mm): on a circle DIST from the ball, 30 deg above the platter."""
    a = 2 * np.pi * k / N * 0.75; e = np.radians(30)
    eye = DIST * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
    z = -eye / np.linalg.norm(eye); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], 1); T[:3, 3] = eye
    return T


def depth_of(T):
    """Depth (0.1 mm) of a ball of RADIUS at the origin seen from T; the rest cut away (as the range cut would)."""
    fx, fy, cx, cy = CAL; u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1)
    R, o = T[:3, :3], T[:3, 3]; d = rays @ R.T
    b = d @ o; a = (d * d).sum(-1); c = o @ o - RADIUS ** 2; disc = b * b - a * c
    t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, 0)
    return np.where(disc > 0, np.round(t * 10), 0).astype(np.uint16)


def make_project(root, sides=1):
    param = os.path.join(root, 'param'); os.makedirs(param, exist_ok=True)
    fx, fy, cx, cy = CAL
    Path(param, 'Pl.bin').write_bytes(struct.pack('<10f', 0, fx, 0, cx, 0, fy, cy, 0, 0, 1))
    proj = os.path.join(root, 'ProjectLive01012026000000'); os.makedirs(os.path.join(proj, 'data'))
    for s in range(sides):
        folder = fmt.new_side_folder(proj, param); cache = os.path.join(folder, 'cache'); stamps, entries = [], []
        for k in range(N):
            name = fmt.frame_name(0, k); T = cam_pose(k); M = T @ F
            d = depth_of(T); ir = np.where(d > 0, 180, 40).astype(np.uint8)
            fmt.write_frame(cache, name, d, ir, M)
            stamps.append(dict(frame=name, t=float(k), ok=True, fit=0.9, rmse=0.5, reloc=False, near=100, far=450, led=False, gain=4, hinted=False))
            entries.append((0, k, M))
        fmt.write_side_report(cache, entries, stamps)
        # the platter a little under the ball's middle: the cut keeps the top of the ball
        Path(folder, 'table_plane.json').write_text(json.dumps(dict(n=[0, 0, 1.0], d=-10.0, margin=1.0, keep_above=True)))
    return proj


@pytest.fixture(scope='module')
def proj():
    root = tempfile.mkdtemp(prefix='pointyoink-timelapse-proj-'); yield make_project(root)
    shutil.rmtree(root, ignore_errors=True)


def probe(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                        'stream=nb_read_frames,width,height,codec_name,pix_fmt:format=duration', '-of', 'json', path],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout); s = j['streams'][0]
    return int(s['nb_read_frames']), float(j['format']['duration']), s


def lit(img): return int((np.asarray(img)[..., :3].astype(int).sum(-1) > sum(TL.BACKGROUND) + 60).sum())


def test_side_plays_its_tracked_frames_and_frames_the_camera(proj):
    s = TL.pick_sides(proj)[0]
    assert len(s.entries) == N and s.plane is not None and np.allclose(s.up(), [0, 0, 1])
    assert np.allclose(s.pose(3), cam_pose(3))
    P = np.random.default_rng(0).normal(size=(2000, 3)) * 20
    cam = TL.ShotCamera(P, [0, 0, 1], [1, 0, 0], 16 / 9); V = cam.view(0.0); Pj = cam.proj(V)
    Q = P @ V[:3, :3].T + V[:3, 3]; assert (Q[:, 2] < 0).all()              # every point in front of the camera
    x = Pj[0, 0] * Q[:, 0] / -Q[:, 2]; y = Pj[1, 1] * Q[:, 1] / -Q[:, 2]
    assert np.abs(x).max() <= 1.0 and np.abs(y).max() <= 1.0 and max(np.abs(x).max(), np.abs(y).max()) > 0.6


def test_watermark_bottom_left():
    w, h = TL.frame_size(480); img = np.zeros((h, w, 3), np.uint8); img[:] = TL.BACKGROUND
    out = TL.Watermark(w, h).apply(img.copy()); diff = np.abs(out.astype(int) - img).sum(-1) > 20
    ys, xs = np.nonzero(diff)
    assert len(ys) > 200
    assert xs.max() < w * 0.3 and ys.min() > h * 0.85                       # bottom-left, small
    assert out.max() < 255 * 0.8                                              # see-through (70%)
    white = np.full((h, w, 3), 255, np.uint8); ow = TL.Watermark(w, h).apply(white.copy())
    assert (np.abs(ow.astype(int) - 255).sum(-1) > 40).sum() > 50            # the shadow shows on white too


@needs_ffmpeg
def test_mp4_grows_and_has_the_length_asked(proj, tmp_path):
    out = str(tmp_path / 'clip.mp4'); still = str(tmp_path / 'still.png'); shares = []
    r = TL.render_timelapse(proj, out, height=240, seconds=2.0, orbit=0.5, fps=10, device='CPU:0',
                            progress=lambda s, t: shares.append(s), still=still)
    assert r['frames'] == 20 and r['fps'] == 10 and r['bytes'] == os.path.getsize(out) > 1000
    n, dur, s = probe(out)
    assert n == 20 and abs(dur - 2.0) < 0.15 and s['codec_name'] == 'h264' and s['pix_fmt'] == 'yuv420p'
    assert (s['width'], s['height']) == (426, 240)
    assert not os.path.exists(out + '.part') and shares[-1] > 0.95 and shares == sorted(shares)
    from PIL import Image
    im = np.asarray(Image.open(still).convert('RGB'))
    assert lit(im[int(240 * 0.85):, :int(426 * 0.3)]) > 50                  # the mark
    assert lit(im[:, int(426 * 0.3):]) > 500                                  # the model


@needs_ffmpeg
def test_gif_camera_and_model_side_by_side(proj, tmp_path):
    out = str(tmp_path / 'clip.gif')
    r = TL.render_timelapse(proj, out, layout='camera+model', height=240, seconds=1.5, orbit=0, fps=8, watermark=False, device='CPU:0')
    from PIL import Image
    im = Image.open(out); assert im.format == 'GIF' and im.n_frames == r['frames'] == 12
    assert abs(im.info['duration'] - 125) <= 10
    frames = []
    for k in range(im.n_frames): im.seek(k); frames.append(np.asarray(im.convert('RGB')))
    left, right = frames[-1][:, :213], frames[-1][:, 213:]
    assert lit(left) > 1000 and lit(right) > 300                             # the IR picture and the model
    assert lit(frames[-1][:, 213:]) > lit(frames[0][:, 213:])                # the model grew


def test_gif_without_ffmpeg_uses_pillow(proj, tmp_path, monkeypatch):
    monkeypatch.setattr(TL, 'ffmpeg', lambda: None)
    assert TL.formats() == ('gif',)
    with pytest.raises(RuntimeError): TL.render_timelapse(proj, str(tmp_path / 'x.mp4'), height=120, seconds=1, device='CPU:0')
    assert not os.listdir(tmp_path)
    out = str(tmp_path / 'clip.gif')
    r = TL.render_timelapse(proj, out, height=180, seconds=1.0, orbit=0.25, device='CPU:0', renderer='cpu')
    from PIL import Image
    im = Image.open(out); assert im.n_frames == r['frames'] == 12 and r['renderer'] == 'cpu'


def test_cancel_leaves_nothing(proj, tmp_path):
    out = str(tmp_path / 'clip.gif'); calls = [0]

    def cancel():
        calls[0] += 1; return calls[0] > 20
    with pytest.raises(TL.Cancelled): TL.render_timelapse(proj, out, height=120, seconds=2, device='CPU:0', cancel=cancel)
    assert os.listdir(tmp_path) == []


def test_default_name_and_sizes(tmp_path):
    p = str(tmp_path / 'ProjectLive01'); os.makedirs(p)
    assert TL.default_path(p, 'mp4').endswith('ProjectLive01_timelapse.mp4')
    Path(p, 'ProjectLive01_timelapse.mp4').write_bytes(b'x')
    assert TL.default_path(p, 'mp4').endswith('ProjectLive01_timelapse_2.mp4')
    assert TL.frame_size(720) == (1280, 720) and TL.frame_size(1080) == (1920, 1080) and TL.frame_size(480) == (852, 480)
    assert TL.spread([100, 300], 40) == [10, 30] and min(TL.spread([1, 1000], 10)) >= 2


def run_cli(args):
    return subprocess.Popen([sys.executable, '-m', 'app2.scan.timelapse'] + args, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)


def test_cli_progress_done_and_sigterm(proj, tmp_path):
    out = str(tmp_path / 'clip.gif')
    p = run_cli([proj, '-o', out, '--size', '480p', '--seconds', '1', '--orbit', '0', '--device', 'CPU:0', '--renderer', 'cpu'])
    lines = p.stdout.read().splitlines(); assert p.wait(120) == 0
    assert any(l.startswith('PROGRESS ') for l in lines)
    done = json.loads(lines[-1][5:]); assert lines[-1].startswith('DONE ') and done['path'] == out and os.path.exists(out)
    assert done['height'] == 480 and done['frames'] == 12
    out2 = str(tmp_path / 'stop.gif')
    p = run_cli([proj, '-o', out2, '--seconds', '30', '--device', 'CPU:0', '--renderer', 'cpu'])
    for line in p.stdout:
        if line.startswith('PROGRESS'): break
    p.send_signal(signal.SIGTERM); rest = p.stdout.read()
    assert p.wait(60) == 2 and 'CANCELLED' in rest
    assert not os.path.exists(out2) and not os.path.exists(out2 + '.part')


def test_coverage_colors_go_red_yellow_green_then_gray():
    import numpy as np
    from app2.scan import timelapse as TL
    pts = np.zeros((3, 3), np.float32); pts[:, 0] = [0, 10, 20]
    c = TL.SurfaceCoverage(pts, 0.5, fade=4.0)
    lv = np.array([0, 1, 2])
    assert c.levels(pts, lv, 1.0).tolist() == [TL.SLOT_RED, TL.SLOT_YELLOW, TL.SLOT_GREEN]
    assert c.levels(pts, lv, 3.0)[2] == 2          # halfway gray
    assert c.levels(pts, lv, 99.0)[2] == 0         # fully gray, the point stays gray
