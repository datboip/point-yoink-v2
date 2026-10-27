# Camera controls and color guidance

Each camera has its own solid header above the full picture. The camera column
keeps the same width when a tile becomes the large view. The corner button makes
IR, Depth, or Color the large top tile. Click it again, or press Esc while a
camera control has focus, to restore the usual order. Pop out opens the three
feeds in a separate window with the same controls and preferences.

The IR header holds Left, Right, and Both. Small tiles use initials, with full
tooltips and accessible names. The number beside them is a fresh observed bed
angle. A question mark means the current angle is unknown; accumulated kept
views are not presented as a fresh angle.

Color's Show marks button opens switches. Depth edge is one preference shared
by IR, Color, the main model's color preview, and the camera window. Its default
is on. A saved explicit choice stays saved. The short key under the cameras
explains the dashed edge and pink missing-depth candidates. Dots and outlines
are drawn after image scaling so their strokes remain readable.

Capture, Object, Surface, model display mode, and the short Live settings lists
use visible choice buttons. Project, calibration, and other open-ended lists
can still use dropdowns.

The shared Build and Edit model viewer also shows its display choices as
buttons. View opens a small panel with projection buttons, a marked-pieces
switch, and Home view actions. Display buttons stay in place when the model
refreshes. Narrow viewports wrap them without moving the model pane.

## Color angle estimate

This experimental setting is off by default in Show marks. It observes white
bed dots in the original color picture and matches their planar arrangement to
the learned dot map. It requires a recent measured bed plane, the scanner's RGB
calibration, enough matched dots, and a clearly better match than the competing
placements. A recent measured bed footprint excludes unrelated background dots.

The readout uses `~` and its tooltip says that the angle is estimated from color.
Keep the camera still. The measured plane expires after the existing framing
window. Losing IR dots briefly does not renew that anchor. A changed bed,
calibration, or a sufficiently changed plane invalidates outstanding results.
Repeated patterns, too few dots, expired anchors, and failed fits return unknown.

Color work runs in its own worker. Only one job may be outstanding. Busy frames
are dropped; a backlog never grows. Calibration and anchor changes are
synchronized. The worker returns an angle and match quality, with no points or
pose matrix. Its result goes only to the camera readout. It does not seed ICP,
move depth overlays, choose kept frames, change a depth image, or enter a volume,
mesh, or saved scan pose. The original color images are not enhanced or altered.

The saved clips demonstrate more angle observations on the Far clip, and some
dropouts and angle bias on Near. These are comparisons with the depth dot
matcher, not an independent angle encoder. They do not validate the estimates
where the reference matcher itself was missing. The saved frames also do not
validate an independent 30 fps color stream or a moved camera.

## Both alignment

Both maps the right IR into left coordinates with each measured pixel's depth.
Small holes can borrow nearby measured depth, preserving the supplied live
tree's hole repair. Regions without such support no longer borrow an arbitrary
part plane for a colored blend. A photometric disagreement check leaves
unmatched or one-camera-only areas as the left image.

Refine Both alignment is another experimental setting, off by default. After
the depth warp, it checks mutual dot matches, rejects outliers, and estimates a
small translation bounded by the observed dot radius. Three agreeing frames
are needed before applying its median. A failed match clears that history.
This changes only the preview. It never changes stereo calibration or depth.

The supplied evidence has no saved right-IR files. Generated stereo tests cover
multiple distances, depth holes, bounded residual offsets, and one-camera-only
features. Hardware alignment, per-dot offsets on real stereo pairs, moving
objects, and lighting differences still need verification. Do not interpret
the generated-pair results as a measured scanner correction.
