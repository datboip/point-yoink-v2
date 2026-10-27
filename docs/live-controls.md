# Live scan controls

The strip above the live views stays visible on both Sides and Settings. It
wraps whole control groups when needed. Its spacing and touch targets follow
the font size. It does not cover a camera picture or the model.

| Control | Shortcut | Use |
| --- | --- | --- |
| Start, Pause, Continue | Space | Record or pause the current side |
| New side | N | Start a different side after pausing |
| Light | L | Request the fill light on or off |
| IR gain | [ and ] | Lower or raise gain |
| IR gain Auto | G | Toggle automatic gain |
| Range Auto | R | Toggle automatic range |
| 3D color | V | Cycle Distance, Coverage, Plain |
| IR camera | I | Cycle Left, Right, Both |

Light sent means the command was sent, without confirmed readback. It is not
proof that the physical light is on. Distance colors need saved views; until
those exist, coverage remains available.

Settings holds Capture, Object, Surface, saving color, bed cuts, camera setup
and tests. Cut beside bed defaults to Off. Off keeps overhangs outside a known
bed outline. A value is how far below the bed to cut beside it, in millimeters.
That cut happens before tracking and saving, so removed depth cannot be restored.
When the bed outline is unknown, the existing full-plane cut still applies.

Automatic Build cuts with a measured bed outline also keep overhangs outside
it. Manual planes without an outline retain their full-plane meaning.
The bad-cut check counts only points the cut would actually remove.

The blue-gray dashed line in Color is the calibrated stereo field at the
measured distance. White dashes show observed depth support. These are separate
from the part box. None of these display lines changes captured points.

Color runs on its own preview worker. Incoming JPEGs wake that worker directly;
depth composition does not hold color behind it. Both Scan and the camera
window reject color older than the picture already shown. Build has a separate
camera-rate footer that disappears after the stream stops.
Color-dot fitting is limited by the existing color synchronization interval,
up to ten jobs per second, while the pictures continue at camera speed.

On a flat bed, an unambiguous dot match anchors the pose before fusion. Depth
must still fit at that pose. Color dots are lifted onto a measured depth plane.
When a known flat bed loses both dot sources, recording waits for dots instead
of fusing a pose that depth cannot verify.
Color can seed a pose only after several agreeing IR/color checks remove their
bias. A color observation expires with its plane and must be close in time to
the depth frame. It does not establish new surface depth.

Bed learning combines color and IR after matching their shared dots. Duplicate
observations count once per frame. A known bed radius bounds color additions.
The full-turn validation gate remains in place; a wider color view alone does
not prove that learning finishes sooner.
