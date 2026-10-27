# SPDX-License-Identifier: AGPL-3.0-or-later
"""The live pictures: IR (green = kept by the range cut, violet = the platter the build will cut, dimmed =
cut below the bed, table dots), depth, and the colour camera with where the depth lands, all 800x600 RGB. Composed on their own thread from the
newest frame only (a slow composition never holds up tracking) and handed to the window as numpy arrays. A sparse-depth estimate is labeled in the picture; the window chips carry the other labels. The drawing math is the Tk
scanner's tick()."""
import io, threading
import numpy as np, cv2
from app2.scan import timing

W, H = 800, 600
OVERLAY_MATRIX = np.eye(3)
PARAM_DIR = None
RGB_CAL = None       # (K, R, T) of the connected scanner's color camera, read from its own param folder (set_rgb_calib)


def set_rgb_calib(param_dir):
    """Read this scanner's color camera calibration; without it the color view shows no depth guides."""
    global RGB_CAL,OVERLAY_MATRIX,PARAM_DIR
    PARAM_DIR=param_dir
    OVERLAY_MATRIX=np.eye(3)
    from fuse import read_rgb_calib
    RGB_CAL = read_rgb_calib(param_dir)
    return RGB_CAL is not None
def distance_note(frame):
    if not frame.zref_mm:return None
    source=getattr(frame,'distance_source',None) or 'center region'
    return source.capitalize()+' %d mm' % round(frame.zref_mm)


YELLOW = (255, 220, 0)


def depth_px_to_rgb(u, v, z_mm, cal):
    """Depth-image pixels at depth z (mm) -> pixel positions in the 800x600 colour view."""
    fx, fy, cx, cy = cal
    P = np.stack([(u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm * np.ones_like(u, dtype=float)], -1)
    K, R, T = RGB_CAL
    q = (P @ R.T + T) @ K.T
    from app2.scan.overlay_alignment import apply
    xy=apply(q[...,:2]/q[...,2:],OVERLAY_MATRIX)
    return xy[...,0],xy[...,1]


def outlined_line(img, a, b, color, w=3):
    cv2.line(img, a, b, (0, 0, 0), w + 4, cv2.LINE_AA); cv2.line(img, a, b, color, w, cv2.LINE_AA)


def cross(img, c, color=YELLOW):
    outlined_line(img, (c[0] - 22, c[1]), (c[0] + 22, c[1]), color); outlined_line(img, (c[0], c[1] - 22), (c[0], c[1] + 22), color)


def scan_area(cal, near, far, depth_size=(W, H)):
    """Fixed range planes of the depth frustum, projected with this scanner's calibration."""
    if RGB_CAL is None or not np.isfinite([near, far]).all() or near <= 0 or far <= near: return None
    width, height = depth_size
    if width <= 1 or height <= 1 or not np.isfinite(cal).all() or cal[0] <= 0 or cal[1] <= 0: return None
    edge = np.array([[0., 0.], [width - 1., 0.], [width - 1., height - 1.], [0., height - 1.], [0., 0.]])
    planes = []
    for z in (near, far):
        rays = np.stack([(edge[:, 0] - cal[2]) / cal[0], (edge[:, 1] - cal[3]) / cal[1], np.ones(len(edge))], 1)
        if np.any((rays * z @ RGB_CAL[1].T + RGB_CAL[2])[:, 2] <= 0): return None
        x, y = depth_px_to_rgb(edge[:, 0], edge[:, 1], np.full(len(edge), z), cal)
        planes.append(np.stack([x, y], 1))
    x, y = depth_px_to_rgb(np.array([width // 2]), np.array([height // 2]), np.array([(near + far) / 2]), cal)
    if not all(np.isfinite(p).all() for p in planes) or not np.isfinite([x[0], y[0]]).all(): return None
    outline = cv2.convexHull(np.concatenate(planes).astype(np.float32)).reshape(-1, 2)
    outline = np.vstack([outline, outline[:1]])
    return dict(near=planes[0], far=planes[1], outline=outline, center=np.array([x[0], y[0]]), range=(near, far))


def coverage_rect(d, keep_share=0.01):
    """Where this depth frame actually has depth, as a depth-pixel rectangle (u0, v0, u1, v1). The stereo pair only
    sees depth where both IR cameras overlap, so this is narrower than the full frame. The thinnest keep_share of
    rows and columns at each edge is ignored so a few stray pixels don't stretch it. None when there is no depth."""
    valid = d > 0
    if valid.sum() < 500: return None
    cols = valid.sum(0).astype(float); rows = valid.sum(1).astype(float)
    def span(counts):
        c = np.cumsum(counts) / counts.sum()
        return float(np.searchsorted(c, keep_share)), float(np.searchsorted(c, 1 - keep_share))
    (u0, u1), (v0, v1) = span(cols), span(rows)
    return (u0, v0, u1, v1) if u1 > u0 and v1 > v0 else None


def live_area(cal, rect, z_mm, depth_size=(W,H)):
    """The depth camera's real coverage rectangle and its center ray, both drawn at the part's distance z_mm, so they
    land on the part in the colour view instead of on a range midpoint (the colour camera sits to one side, so the
    distance matters)."""
    if RGB_CAL is None or rect is None or z_mm is None or not np.isfinite(z_mm) or z_mm <= 0: return None
    width,height=depth_size
    u0, v0, u1, v1 = rect
    edge = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]], float)
    x, y = depth_px_to_rgb(edge[:, 0], edge[:, 1], np.full(len(edge), z_mm), cal)
    # the IR view's + marks the depth picture's middle, so the colour view's center is that same pixel (not the
    # principal point, which sits about 100 px off center on this scanner's Far calibration)
    cx, cy = depth_px_to_rgb(np.array([width // 2]), np.array([height // 2]), np.array([z_mm]), cal)
    outline = np.stack([x, y], 1)
    if not np.isfinite(outline).all() or not np.isfinite([cx[0], cy[0]]).all(): return None
    return dict(outline=outline, center=np.array([cx[0], cy[0]]), live=True)


def closed_hull(points):
    if not np.isfinite(points).all():return None
    hull=cv2.convexHull(np.asarray(points,np.float32)).reshape(-1,2)
    return np.vstack([hull,hull[:1]])


def edge_depth(depth,cols,rows,fallback,reach=None):
    """Depth in mm at each edge point, read from the frame just inside it (median of a short run of pixels toward
    the middle), else fallback. The IR and color cameras sit apart, so an edge drawn at one distance lands in the
    wrong place on a part that comes closer or goes farther (seen live: a tilted part ran past the drawn IR edge)."""
    z=np.full(len(cols),float(fallback))
    if depth is None or not len(cols):return z
    height,width=depth.shape;reach=reach or max(4,width//80)
    c=np.rint(cols).astype(int);r=np.clip(np.rint(rows).astype(int),0,height-1)
    step=np.where(c<width//2,1,-1)                       # toward the middle
    run=np.clip(c[:,None]+step[:,None]*np.arange(reach+1)[None,:],0,width-1)
    vals=depth[r[:,None],run].astype(np.float64);vals[vals<=0]=np.nan
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(vals,axis=1)
    ok=np.isfinite(med);z[ok]=med[ok]*.1   # one vectorized pass: a median per edge point in a loop ate the color thread
    return z


def stereo_limit(stereo,cal,shape,distance,plane=None,depth=None):
    """Calibrated stereo overlap, distinct from observed depth support. With the depth frame, each row's edge sits
    at the depth found there, so it follows the part instead of one distance."""
    if stereo is None or distance is None or not np.isfinite(distance) or distance<=0:return None
    height,width=shape
    rows=np.arange(height,dtype=float)
    z=np.full(height,distance)
    if plane is not None:
        n,off=plane[:2]
        den=(rows-cal[3])*n[1]/cal[1]+n[2]
        with np.errstate(divide='ignore',invalid='ignore'):z=-off/den
    valid=np.isfinite(z)&(z>0)
    if not valid.all():return None
    shift=stereo.right(np.column_stack([np.zeros(height),rows]),z)[:,0]
    left=np.maximum(0,-shift);right=np.minimum(width-1,width-1-shift)
    if np.any(left>=right):return None
    edge=np.vstack([np.column_stack([left,rows]),np.column_stack([right[::-1],rows[::-1]]),[[left[0],rows[0]]]])
    distances=np.r_[z,z[::-1],z[:1]]
    if depth is not None:distances=edge_depth(depth,edge[:,0],edge[:,1],distance)
    return np.column_stack(depth_px_to_rgb(edge[:,0],edge[:,1],distances,cal))


def project_outline(cal,edge,span):
    projected=[]
    for z in span:
        x,y=depth_px_to_rgb(edge[:,0],edge[:,1],np.full(len(edge),z),cal)
        projected.append(np.column_stack([x,y]))
    return closed_hull(np.concatenate(projected))


def part_area(cal,frame):
    """Conservative color bounds of the full connected depth part, including parallax."""
    rect=frame.get('rect');z=frame.get('depth_mm')
    area=live_area(cal,rect,z,frame['depth_size'])
    if area is None:return None
    span=frame.get('observed_range_mm') or [z,z]
    edge=np.asarray([[rect[0],rect[1]],[rect[2],rect[1]],[rect[2],rect[3]],[rect[0],rect[3]]])
    projected=[]
    for distance in span:
        x,y=depth_px_to_rgb(edge[:,0],edge[:,1],np.full(len(edge),distance),cal);projected.append(np.column_stack([x,y]))
    pts=np.concatenate(projected)
    if not np.isfinite(pts).all():return None
    lo,hi=pts.min(0),pts.max(0)
    area['outline']=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]],[lo[0],lo[1]]])
    center=frame.get('center_px')
    if center is not None:
        x,y=depth_px_to_rgb(np.array([center[0]]),np.array([center[1]]),np.array([z]),cal);area['part_center']=np.array([x[0],y[0]])
    observed=frame.get('depth_outline')
    if observed is not None:
        area['depth_outline']=project_outline(cal,np.asarray(observed),span)
    return area


def dot_samples(d, cell=6):
    """One depth sample per cell on a straight grid (the user prefers the aligned rows to a random speckle)."""
    v,u=np.nonzero((d>0)[::cell,::cell]);return v*cell,u*cell



def depth_guides(rgb, d, cal, z_ref, dot, area=None, bands=None, cut=None, vectors=True):
    """Connected part box and dot, separate scanner aim, and a faint depth-view border."""
    if RGB_CAL is None:return rgb
    h,w=rgb.shape[:2]
    if vectors and area is not None and area.get('depth_outline') is not None:
        outline=np.rint(area['depth_outline']).astype(np.int32)
        # no dimming outside: the dotted edge alone marks it
        dash=max(2,round(h/120))                               # a light dotted edge; the tilt is the real camera offset
        for p0,p1 in zip(outline,np.roll(outline,-1,0)):
            steps=max(1,int(np.hypot(*(p1-p0))//(2*dash)))
            for k in range(2*steps):                           # light and dark dashes in turn: one of them shows on any background
                a0=p0+(p1-p0)*k/(2*steps);a1=p0+(p1-p0)*(k+1)/(2*steps)
                cv2.line(rgb,tuple(np.rint(a0).astype(int)),tuple(np.rint(a1).astype(int)),(235,238,242) if k%2==0 else (30,34,40),1,cv2.LINE_8)
    vv,uu=dot_samples(d)
    if len(uu):
        x,y=depth_px_to_rgb(uu.astype(float),vv.astype(float),d[vv,uu].astype(float)*.1,cal)
        ok=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<w-1)&(y>=0)&(y<h-1)
        xi,yi=x[ok].astype(int),y[ok].astype(int)
        z=d[vv,uu][ok].astype(float)*.1
        if bands is not None:
            from app2.scan.range_guide import BAND_RGB,distance_levels
            profile=bands if isinstance(bands,dict) else dict(observed_distance_mm=bands)
            colors=np.array([BAND_RGB['off'],BAND_RGB['ok'],BAND_RGB['best']],np.uint8)[distance_levels(z,profile)]
        else:colors=np.tile(np.asarray(dot,np.uint8),(len(z),1))
        if cut is not None:
            removed=(z<cut[0])|(z>cut[1]);colors[removed]=REMOVED_RGB
        rgb[yi,xi]=colors;rgb[yi+1,xi]=colors           # short vertical marks on a grid: the look the user prefers
    if vectors and area is not None:
        pts=np.rint(area['outline']).astype(np.int32);color=tuple(area.get('color',YELLOW))
        cv2.polylines(rgb,[pts],True,color,1,cv2.LINE_AA)
        # Sampling marks scale with the caption font, not with the size of the box.
        height=cv2.getTextSize('Aim',cv2.FONT_HERSHEY_SIMPLEX,.5*h/H,1)[0][1]
        arm=max(1,round(height/2));c=tuple(np.rint(area['center']).astype(int))
        cv2.line(rgb,(c[0]-arm,c[1]),(c[0]+arm,c[1]),color,1,cv2.LINE_AA)
        cv2.line(rgb,(c[0],c[1]-arm),(c[0],c[1]+arm),color,1,cv2.LINE_AA)
        if SHOW_PART_CENTER and area.get('part_center') is not None:
            center=tuple(np.rint(area['part_center']).astype(int));box=pts.max(0)-pts.min(0)
            # Close enough (within a tenth of the box) reads as centered: one ring around the plus, no second mark.
            if np.hypot(center[0]-c[0],center[1]-c[1])<=.1*min(box):cv2.circle(rgb,c,arm+max(1,round(height/4)),color,1,cv2.LINE_AA)
            else:                                              # off center: a thin line from the part's middle to the aim shows which way to move
                cv2.line(rgb,center,c,color,1,cv2.LINE_AA);cv2.circle(rgb,center,max(1,round(height/4)),color,-1,cv2.LINE_AA)
        # no words on the camera views: the box color carries the state, hints live outside the picture
    return rgb


SHOW_PART_CENTER = False                               # the part-center dot and its line to the aim; off until the Show menu (round 27)
REMOVED_RGB = (140, 146, 156)                          # depth the Range leaves out: plain gray, not a band color
CUT_DIM = np.array([40, 60, 90], np.uint16)             # what Cut below bed took: dimmed to a third, a little blue


DEPTH_TINT = ([6, 10, 6], [0, 70, 0])   # IR pixels with depth: x/10 per channel, plus this


def ir_view(d, irl, cal, lost_now, geo, dots, cut=None, aim=True, outside_below_mm=None):
    """IR with the range cut tinted (green, red while lost), the platter the build will cut in violet with its rim as a
    dashed ring, what Cut below bed took (cut: flat pixel indices) dimmed, the table dots circled (green = matched to
    the bed, grey = not) and the view's centre."""
    H,W=irl.shape
    ir = cv2.cvtColor(np.ascontiguousarray(irl), cv2.COLOR_GRAY2RGB)
    if cut is not None and len(cut):
        f = ir.reshape(-1, 3); f[cut] = (f[cut].astype(np.uint16) // 3 + CUT_DIM // 3).astype(np.uint8)
    keep = d > 0
    tint = DEPTH_TINT
    ir[keep] = (ir[keep].astype(np.uint16) * tint[0] // 10 + tint[1]).clip(0, 255).astype(np.uint8)
    if geo is not None:
        # Only pixels with depth at or under it are tinted, and the platter's rim is drawn as a ring. (Filling the whole
        # disc where there is no depth also painted every part of the part that gave no depth: seen live.)
        nc, oc, cen, rad_ = geo
        rx = (np.arange(W, dtype=np.float32) - cal[2]) / cal[0]; ry = (np.arange(H, dtype=np.float32) - cal[3]) / cal[1]
        den = rx[None, :] * nc[0] + ry[:, None] * nc[1] + nc[2]             # n . ray, ray = (rx, ry, 1)
        from app2.scan.bedcut import LOW_MM
        on = keep & (d.astype(np.float32) * 0.1 * den + oc < LOW_MM)   # the same height the cut itself uses: purple is what goes
        if cen is not None and rad_ is not None and on.any():
            # Only over the bed, like the cut itself (bedcut.py): a part hanging past the rim below the bed's height stays.
            from app2.scan.bedcut import BedCut, RIM_MM, OUTSIDE_BELOW_MM
            v, u = np.nonzero(on); z = d[v, u].astype(np.float64) * 0.1
            P = np.column_stack([rx[u] * z, ry[v] * z, z])
            over = BedCut._sd(P, np.asarray(nc, np.float64), np.asarray(cen, np.float64), rad_) < RIM_MM
            threshold=OUTSIDE_BELOW_MM if outside_below_mm is None else outside_below_mm
            low = (P @ np.asarray(nc, np.float64) + oc < -threshold) if threshold>0 else np.zeros(len(P),bool)
            on[v[~(over | low)], u[~(over | low)]] = False
        ir[on] = (ir[on].astype(np.uint16) // 2 + np.array([80, 0, 110], np.uint16)).clip(0, 255).astype(np.uint8)
        if cen is not None and rad_ is not None:                          # the platter's centre, camera frame
            if hasattr(rad_, "ring"):                                      # a rectangle, or a circle off the spin centre
                ring = rad_.ring(nc, cen, 120); ring = np.vstack([ring, ring[:1]])
            else:
                e1 = np.cross(nc, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nc, e1)
                ang = np.linspace(0, 2 * np.pi, 121)
                ring = cen + rad_ * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
            z = np.maximum(ring[:, 2], 1e-6); px = np.stack([ring[:, 0] / z * cal[0] + cal[2], ring[:, 1] / z * cal[1] + cal[3]], 1)
            for k in range(0, 120, 2):                                    # dashed, and only what is in front of the camera
                if ring[k, 2] > 20 and ring[k + 1, 2] > 20 and np.abs(px[k]).max() < 4000 and np.abs(px[k + 1]).max() < 4000:
                    cv2.line(ir, tuple(int(v) for v in px[k]), tuple(int(v) for v in px[k + 1]), (200, 90, 255), 2, cv2.LINE_AA)
    if dots is not None:
        for (cx_, cy_, r_), m_ in zip(dots[0], dots[1]):
            cv2.circle(ir, (int(cx_), int(cy_)), int(1.6 * r_ + 4), (60, 230, 60) if m_ else (150, 150, 150), 2, cv2.LINE_AA)
    if aim:cross(ir, (W // 2, H // 2))
    return ir


def depth_view(d):
    """Depth in turbo colours, near = warm, stretched over the 2nd to 98th percentile of the kept depth."""
    dv = np.zeros((*d.shape,3), np.uint8); nz = d > 0
    if nz.any():
        lo, hi = np.percentile(d[nz], [2, 98])
        v = np.clip(255 - (d[nz].astype(np.float32) - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8)
        dv[nz] = cv2.applyColorMap(v.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]
    return dv


_predecoded = (None, None)   # (the JPEG bytes object, its decoded view) from the camera's own reader thread


def predecode(jpeg):
    """Decode as the frame arrives, on the color reader thread (OpenCV lets go of the GIL), so the composer only
    picks up the result instead of decoding in line behind depth work."""
    global _predecoded
    try: _predecoded = (jpeg, decode_colour(jpeg))
    except Exception: pass


def decoded(jpeg):
    key, image = _predecoded
    return image if key is jpeg else decode_colour(jpeg)


def decode_colour(jpeg):
    """The color camera's JPEG resized to the calibrated 800x600 view. The JPEG decoder scales by powers of two while
    decoding, never below the view size (2000x1500 decodes at 1000x750, about 6x faster than full size)."""
    # OpenCV decodes at half size straight from the JPEG and lets go of the GIL while it works (PIL held it, and
    # the window thread starved color to about 11 of the camera's 24 frames a second)
    raw = np.frombuffer(jpeg, np.uint8)
    im = cv2.imdecode(raw, cv2.IMREAD_REDUCED_COLOR_2)
    if im is None or im.shape[1] < W or im.shape[0] < H: im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if im is None: raise ValueError("color frame is not a JPEG")
    return cv2.cvtColor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


class LiveView:
    """One composed set of live pictures for the window (HxWx3 uint8 each; colour None before the first colour frame).
    zref_mm: the part's distance, smoothed. lost_now: tracking lost on this frame. platter: the IR picture shows the
    platter's tint (its legend is shown only then). cut: pixels Cut below bed took from this frame (0: none)."""
    __slots__ = ("ir", "depth", "colour", "zref_mm", "lost_now", "seq", "platter", "colour_only", "cut", "distance_source", "colour_clean", "colour_guides", "ir_guides", "right_guides", "ir_right", "ir_both", "color_angle", "bed_angle", "color_token")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))


class Composer:
    """Composes the live pictures on its own thread. post() hands in the newest frame (an older waiting one is dropped);
    emit('live', LiveView) goes to the window. Keeps the smoothed part distance and the last colour picture.
    colour_fn: returns the newest colour JPEG. Depth (and the IR packed with it) comes at about 3 fps, but the colour
    camera runs at 30: between depth frames the colour picture is redrawn on its own, up to COLOUR_HZ times a second,
    with the last depth frame's outline and the last IR and depth pictures (the user: "the video is at least realtime
    on the scanner")."""
    COLOUR_HZ = 60.0   # how often to look for a new color frame; a 1/30 s wait plus the decode could never keep up with 30 fps
    COVER_S = 3.0      # the scan box is where depth showed up over this many seconds: steady, but follows a new range

    def __init__(self, cal, emit, colour_fn=None):
        self.cal = cal; self.emit = emit; self.zref = None; self.rgb = None; self.jpeg = None
        self._rgb_packet=(None,None,0)
        self.distance_source="center region"
        self.colour_fn = colour_fn; self.last = None       # (d, lost_now, ir picture, depth picture, seq, platter, cut)
        self.area = None; self.scan_geometry = None; self.cover = []; self.live = None;self.reference_plane=None
        self.framing=None;self.guide_depth=None;self.dot_data=None;self.color_missing=None;self.rgb_depth=None
        from app2.scan.color_surface import ColorSurface
        self.color_surface=ColorSurface()
        from app2.scan.color_tracking import ColorObserver
        from app2.scan import settings
        self.color_observer=ColorObserver(self.COVER_S)
        self.color_angle_enabled=bool(settings.load().get('color_angle',False))
        self.color_angle=None
        self.bed_color_observer=None;self.bed_angle=None
        self.lockstep=False                                   # recording: color shows the frame matched to the depth frame, with it
        self.ir_mode='Left'                                   # Left, Right or Both: only the shown IR picture is built (Both's blend cost 26 ms a frame)
        from app2.scan.stereo_view import Stereo
        from app2.scan.depth_footprint import DepthFootprint
        self.stereo=Stereo.load(PARAM_DIR,cal) if PARAM_DIR else None
        self.footprint=DepthFootprint(self.COVER_S)
        self.ir_right=None;self.ir_both=None;self.ir_guides=None;self.right_guides=None
        self.box = None; self.cv = threading.Condition(); self.running = True
        self.color_cv=threading.Condition();self.color_ready=False
        source=getattr(colour_fn,'__self__',None)
        self.color_source=source
        subscribe=getattr(source,'subscribe_color',None)
        self.unsubscribe_color=subscribe(self.wake_color) if subscribe else None
        self.color_thread=None
        if colour_fn:
            self.color_thread=threading.Thread(target=self._run_color,name='color-preview',daemon=True)
            self.color_thread.start()
        self.thread = threading.Thread(target=self._run, name="composer", daemon=True); self.thread.start()

    def wake_color(self):
        with self.color_cv:self.color_ready=True;self.color_cv.notify()

    def post(self, job):
        with self.cv: self.box = job; self.cv.notify()

    def set_scan_geometry(self, near, far, scan_mode='near'):
        """Range/mode changes update geometry once; scene depth never moves its border."""
        key = (float(near), float(far), scan_mode, tuple(self.cal),
               None if RGB_CAL is None else tuple(np.asarray(v).tobytes() for v in RGB_CAL))
        with self.cv:
            if key != self.scan_geometry:
                self.area = scan_area(self.cal, near, far); self.scan_geometry = key
            from app2.scan.range_guide import measured_profile
            self.dot_bands=measured_profile(scan_mode);self.range_cut=(near,far)

    def stop(self):
        with self.cv: self.running = False; self.cv.notify()
        if self.unsubscribe_color:self.unsubscribe_color();self.unsubscribe_color=None
        self.wake_color()
        if self.color_thread is not None:self.color_thread.join(2)
        self.color_observer.close()

    def _run(self):
        while True:
            with self.cv:
                while self.box is None and self.running:self.cv.wait()
                if not self.running: return
                job, self.box = self.box, None
            try:
                if job is not None:
                    timing.mark('compose_start'); self.emit("live", self.compose(**job)); timing.mark('compose_end')
                    self.wake_color()
            except Exception as e: print("preview:", e, flush=True)

    def _run_color(self):
        while True:
            with self.color_cv:
                if not self.color_ready and self.running:
                    self.color_cv.wait(None if self.unsubscribe_color else 1.0/self.COLOUR_HZ)
                if not self.running:return
                self.color_ready=False
            try:
                v=self.colour_only()
                if v is not None:self.emit('live',v);timing.mark('color_emitted')
            except Exception as e:print('color preview:',e,flush=True)

    def update_cover(self, d, now=None):
        """Remember where this frame had depth; the live box is the union over the last COVER_S seconds, drawn at
        the part's smoothed distance. Starts over when the calibration changes."""
        import time
        now = time.monotonic() if now is None else now
        cal = tuple(self.cal)
        if getattr(self, 'cover_cal', None) != cal: self.cover = []; self.cover_cal = cal
        height,width=d.shape
        rect = coverage_rect(d)
        if rect is not None: self.cover.append((now, rect))
        self.cover = [(t, r) for t, r in self.cover if now - t <= self.COVER_S]
        if not self.cover:
            self.live=live_area(self.cal,(0,0,width-1,height-1),self.zref,(width,height))
            if self.live is not None:self.live['estimated']=True
            return
        r = np.array([r for _, r in self.cover])
        rect=(r[:,0].min(),r[:,1].min(),r[:,2].max(),r[:,3].max())
        sparse=not (rect[0]<=width//2<=rect[2] and rect[1]<=height//2<=rect[3])
        if sparse:rect=(0,0,width-1,height-1)
        self.live=live_area(self.cal,rect,self.zref,(width,height))
        if self.live is not None:self.live['estimated']=sparse

    def guide_area(self):
        frame=getattr(self,'framing',None)
        if frame and frame.get('source')=='part':
            if frame.get('rect') is None:return None
            area=part_area(self.cal,frame)
            if area is not None:
                from app2.scan.range_guide import COLORS
                state=frame.get('state');quality=frame.get('quality','unmeasured')
                name=quality if state=='framed' else 'sparse depth' if state in ('off center','partly out of view','outside Range','missing part depth','possible missing part depth') else state
                color=COLORS.get(name,COLORS['unmeasured']);area['color']=tuple(int(color[k:k+2],16) for k in (1,3,5))

            return area
        live = getattr(self, 'live', None)
        area=live if live is not None else self.area
        if area is not None:area=dict(area)
        return area

    def vector_payload(self,picture=None):
        picture=self._rgb_packet if picture is None else picture
        area=self.guide_area()
        calm=bool((self.framing or {}).get('finding_bed'))
        guides=dict(size=[W,H]) if area is None or calm else dict(area,size=[W,H],aim=area['center'],box=area['outline'])
        if self.dot_data is not None:guides.update(self.dot_data)   # depth dots need no bed: shown from the first frame
        if not calm and self.color_missing is not None:guides['missing_xy']=self.color_missing
        if guides is not None and self.zref and RGB_CAL is not None:
            shape=self.last[0].shape if self.last is not None else (H,W)
            depth=self.last[0] if self.last is not None else None
            guides['stereo_limit']=None   # not drawn any more (depth thins out well inside it): not worked out either
            edge=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1],[0,0]],float)
            span=(self.framing or {}).get('observed_range_mm') or [self.zref,self.zref]
            guides['left_fov']=project_outline(self.cal,edge,span)
            if self.stereo is not None:
                pts=[]
                for z in span:
                    q=self.stereo.left(edge,z);x,y=depth_px_to_rgb(q[:,0],q[:,1],np.full(len(q),z),self.cal)
                    pts.append(np.column_stack([x,y]))
                guides['right_fov']=closed_hull(np.concatenate(pts))
            # No sensor rectangle fallback for the measured depth edge.
            observed=(self.ir_guides or {}).get('depth_outline')
            if observed is not None and not calm:
                o=np.asarray(observed,float);guides['depth_outline']=np.column_stack(depth_px_to_rgb(o[:,0],o[:,1],edge_depth(depth,o[:,0],o[:,1],self.zref),self.cal))
            else:guides['depth_outline']=None
        return dict(color_angle=self.color_angle,bed_angle=self.bed_angle,colour_clean=picture[1],color_token=picture[2],colour_guides=guides,ir_guides=self.ir_guides,
                    right_guides=self.right_guides,ir_right=self.ir_right,ir_both=self.ir_both)

    def _redot(self):
        """The depth dots for the current picture from its paired depth frame (rgb_depth): the bands and the Range cut
        as compose draws them. Called when a picture arrives between depth frames."""
        sample=getattr(self,'rgb_depth',None)
        if sample is None or self.rgb is None or RGB_CAL is None or self.dot_data is None or self.dot_data.get('dots_random'):return
        vv,uu=dot_samples(sample,cell=3);z=sample[vv,uu].astype(float)*.1;x,y=depth_px_to_rgb(uu,vv,z,self.cal)
        from app2.scan.range_guide import distance_levels
        bands=getattr(self,'dot_bands',None) or dict(observed_distance_mm=[0,1000]);levels=distance_levels(z,bands)
        cut_range=getattr(self,'range_cut',None);removed=np.zeros(len(z),bool) if cut_range is None else (z<cut_range[0])|(z>cut_range[1])
        self.dot_data=dict(self.dot_data,dots_xy=np.column_stack([x,y]),dots_uv=np.column_stack([uu,vv]),dots_levels=levels,dots_removed=removed)

    def colour_only(self):
        """A new colour frame between depth frames: only the colour picture is redrawn (None when nothing is new)."""
        if self.lockstep: return None                         # color moves with the depth frames while recording (compose draws it)
        latest=getattr(self.color_source,'latest_color',lambda:None)()
        jpeg = latest['jpeg'] if latest is not None else self.colour_fn() if self.colour_fn else None
        if jpeg is None or jpeg is self.jpeg or self.last is None: return None
        timing.mark('color_decode_start')
        try:
            image=decoded(jpeg)
            self.rgb=image;self.jpeg=jpeg
            picture=self._rgb_packet=(jpeg,image,self._rgb_packet[2]+1)
        except Exception: return None
        timing.mark('color_decoded')
        self.rgb_depth=self.guide_depth if self.guide_depth is not None else self.last[0]   # this picture pairs with the newest depth
        self._redot()
        if self.bed_color_observer is not None:self.bed_angle=self.bed_color_observer.offer(self.rgb,now=None if latest is None else latest['t'])
        self.color_angle=self.color_observer.offer(self.rgb) if self.color_angle_enabled else None
        d, lost_now, ir, dv, seq, platter, cut = self.last
        rgbv = picture[1]
        return LiveView(ir=ir, depth=dv, colour=rgbv, zref_mm=self.zref, lost_now=lost_now, seq=seq, platter=platter,
                        colour_only=True, cut=cut,distance_source=self.distance_source,**self.vector_payload(picture))

    def compose(self, d, irl, lost_now, geo=None, dots=None, jpeg=None, seq=0, cut=None,framing=None,guide_depth=None,irr=None):
        self.framing=framing;self.guide_depth=guide_depth
        ir = ir_view(d, irl, self.cal, lost_now, geo, dots, cut,aim=False,outside_below_mm=getattr(self,'outside_below_mm',None))
        from app2.scan.range_guide import center_distance
        height,width=d.shape
        z_now=center_distance(d)
        plane=geo[:2] if geo is not None else getattr(self,"reference_plane",None)
        self.distance_source="center region" if z_now is not None else "last depth"
        if not d[height//2,width//2] and plane is not None:
            n,off=plane;ray=np.array([(width//2-self.cal[2])/self.cal[0],(height//2-self.cal[3])/self.cal[1],1.])
            den=float(np.asarray(n)@ray)
            if den:z_now=float(-off/den);self.distance_source="mat estimate"
        if z_now is not None and np.isfinite(z_now) and z_now>0:
            self.zref=z_now if self.zref is None else .85*self.zref+.15*z_now
        if framing and framing.get('source')=='part' and framing.get('depth_mm') is not None:
            self.zref=framing['depth_mm'];self.distance_source='part region'
        dv = depth_view(d)
        self.update_cover(d)
        import time
        scene=framing or dict(range_mm=None)
        outline=scene.get('depth_outline')
        if outline is None:outline=self.footprint.update(d,scene,time.monotonic())
        self.ir_guides=dict(size=[width,height],aim=[width//2,height//2],depth_outline=outline)
        if scene.get('finding_bed'):self.ir_guides=dict(size=[width,height])
        mode=self.ir_mode
        self.ir_right=None if irr is None or mode!='Right' else cv2.cvtColor(np.ascontiguousarray(irr),cv2.COLOR_GRAY2RGB)
        self.right_guides=None;self.ir_both=None
        if irr is not None and self.stereo is not None and self.zref and mode!='Left':
            edge=None if outline is None else self.stereo.right(outline,self.zref)
            self.right_guides=dict(size=[width,height],aim=self.stereo.right([width//2,height//2],self.zref),depth_outline=edge)
            if mode=='Both':self.ir_both=self.stereo.blend(irl,irr,d,self.zref)
            if self.ir_right is not None:                          # same green depth tint as Left, moved into the right picture
                from app2.scan.stereo_view import right_depth_mask
                keep=right_depth_mask(self.stereo,d);b=self.ir_right;b[keep]=(b[keep].astype(np.uint16)*np.array(DEPTH_TINT[0],np.uint16)//10+np.array(DEPTH_TINT[1],np.uint16)).clip(0,255).astype(np.uint8)
        if (self.colour_fn is None or self.lockstep) and jpeg is not None and jpeg is not self.jpeg:
            try:
                self.rgb = decoded(jpeg); self.jpeg = jpeg; self.rgb_depth = None   # a new picture: pairs with this frame's depth
                self._rgb_packet=(jpeg,self.rgb,self._rgb_packet[2]+1)
                if self.lockstep and self.bed_color_observer is not None:self.bed_angle=self.bed_color_observer.offer(self.rgb)
            except Exception: pass
        rgbv = None
        if self.rgb is not None:
            rgbv = self.rgb
        self.dot_data=None;self.color_missing=None
        if self.rgb is not None and RGB_CAL is not None:
            # 2026-10-25: the dots belong to the picture they are drawn on. The picture arrives at 2 to 2.5 fps, depth
            # at 3, so a picture keeps the depth frame it came with until the next picture ("the overlay matches the
            # live view instead of the picture"): projecting the newest depth onto an older picture slid the dots off
            # the part while the table turned.
            if getattr(self,'rgb_depth',None) is None:self.rgb_depth=(self.guide_depth if self.guide_depth is not None else d)
            sample=self.rgb_depth
            vv,uu=dot_samples(sample,cell=3)
            if scene.get('range_test'):
                rng=np.random.default_rng(0);vv=np.arange(0,height,3)[:,None]+rng.integers(0,3,(len(range(0,height,3)),len(range(0,width,3))));uu=np.arange(0,width,3)[None,:]+rng.integers(0,3,vv.shape);vv=vv.ravel();uu=uu.ravel();valid=(vv<height)&(uu<width);vv,uu=vv[valid],uu[valid];valid=sample[vv,uu]>0;vv,uu=vv[valid],uu[valid]
            z=sample[vv,uu].astype(float)*.1;x,y=depth_px_to_rgb(uu,vv,z,self.cal)
            from app2.scan.range_guide import distance_levels
            bands=getattr(self,'dot_bands',None) or dict(observed_distance_mm=[0,1000]);levels=distance_levels(z,bands)
            cut_range=getattr(self,'range_cut',None);removed=np.zeros(len(z),bool) if cut_range is None else (z<cut_range[0])|(z>cut_range[1])
            self.dot_data=dict(dots_xy=np.column_stack([x,y]),dots_uv=np.column_stack([uu,vv]),dots_levels=levels,dots_removed=removed,dots_random=bool(scene.get('range_test')))
            part=sample>0 if self.guide_depth is not None and scene.get('source')=='part' else None
            missing=self.color_surface.update(self.rgb,d,part,self.cal,__import__('time').monotonic(),scene.get('scene'))
            if missing is not None:
                vv,uu=dot_samples(missing['mask'].astype(np.uint16));self.color_missing=np.column_stack([uu,vv])
        self.color_angle=self.color_observer.offer(self.rgb) if self.color_angle_enabled and self.rgb is not None else None
        nc = 0 if cut is None else len(cut)
        self.last = (d, lost_now, ir, dv, seq, geo is not None, nc)
        if (self.colour_fn is None or self.lockstep) and self._rgb_packet[1] is not self.rgb:
            self._rgb_packet=(self.jpeg,self.rgb,self._rgb_packet[2]+1)
        picture=self._rgb_packet
        return LiveView(ir=ir, depth=dv, colour=picture[1], zref_mm=self.zref, lost_now=lost_now, seq=seq, platter=geo is not None,
                        cut=nc,distance_source=self.distance_source,**self.vector_payload(picture))
