# Camera checks and scene setup

## Choose the object setting

**Object** has two choices: **Small part** for detail and **Large part** for a wider camera view. Saved projects still use the keys `near` and `far`. Choosing an object does not command a physical camera switch. The app checks the actual stream against both saved camera calibrations.

Open **Settings**, then **Camera setup** to choose calibration files from a scanner export. These are setup files, not a scan or a model. The section opens when the active calibration is missing or does not match. A calibration discovered in an old project is only a viewing bootstrap; it cannot silently replace a saved camera calibration.

The camera check uses repeated dot fits or independent color-circle agreement. It checks again while idle, using the existing three-second guide window, and after a distance or dot-size change larger than the measured scene spread. The notice gives the matched object setting in one steady line. Distance, dot count and fit error go to Log or a tooltip. Each result and its evidence go to Log. An uncertain result does not prove a different camera.

When the frames match the other setting, click **Use Small part** or **Use Large part**. Capture stays paused. A side with saved frames is saved and a new side is prepared before changing the camera calibration. A confirmed change during recording pauses the side and asks for the matching setting. A result from the scene before a large move is discarded.

## Read the distance footer

The distance footer sits below the image and the 3D viewport. Rewind takes priority. On very short windows the footer hides before it can cover the picture.

Distance bands use one shared policy and palette for the bar, projected dots and Capture distance model. Green is Best, yellow is OK, and red is outside the supported distances. With no saved Range test, Best is the middle 40% of the shipped observed span and OK covers the remaining 30% at each end. A saved Range test takes priority. This is a capture guide, not a dimensional accuracy rating.

The fixed axis shows the mode span plus half that span on each side. Histogram columns split source-bin counts by overlap, so narrow display columns do not miss source bins. Faded bars are outside Range. Gray in measured distance bands means an unobserved gap; gray on a model means no saved surface support within the voxel sampling bound. The model key stays visible in Capture distance mode.

## Follow the part through a turn

The part box follows depth above the measured bed. Connected overhangs stay in the box. Outward bounds and depth extrema are retained for the side, so noise cannot make the box shrink across a rotating tip. The center and status use the existing observation window. All color projections use both observed depth extrema to include parallax. A new side or a large scene move resets the span.

**Use part Range** applies the whole side's observed span, rounded outward to the existing Range step, and turns off Auto range. **Undo** restores the previous cut and Auto state. Both require capture to be paused. The former frozen-box control and its keyboard shortcut are gone. Old saved fields are ignored on load and removed when the side is saved again.

Use **Show** for color marks. The part-center mark and edges start off. The plus is the aim; the optional center mark becomes a ring when close enough. The framing explanation is available only on the status text. No framing chip or blanket camera tooltip covers the image. Vector strokes are painted at screen resolution in all camera presentations.

The faint IR depth edge and optional color edge use the same recent observed support. Check dark or shiny regions inside that edge before assuming they are outside the depth view. Interior holes and enclosed dark extensions get separate hints. These are observation cues; unseen geometry is not segmented or reconstructed from a dark IR patch.

The IR selector offers Left, Right and Both. Missing right files in an old clip remain unavailable. Both uses the left view, aligns valid pixels using depth, and uses a plane for holes; it is not a wider stitch. Test real paired alignment on the scanner before relying on empty-region agreement. The turn ring uses learned-bed poses and kept frames only. Saved angular gaps stay gaps until another kept view reaches them.

## Check the scene

The single framing status line gives the most useful hint. Its tooltip includes part point count, fill, the last camera-check dot count when available, and details for each measured problem: off-center part, clipped image edge, sparse depth, excess background inside Range, or mostly black or clipped IR. Missing inputs are shown as unknown. With a measured bed, Auto range receives foreground depth rather than table and background depth.

A steady scene can show a slow-movement hint for handheld scanning. This is a conditional hint, not a physical motion or turntable detector. Auto gain keeps its initial IR estimate and its existing turn-based search. A still-part gain sweep needs paired same-scene measurements at different gains; the supplied fixed-gain clips do not provide that evidence.

## Stream checks

A warning needs at least four near-empty depth events or delivery gaps with a consistent phase over three periods. Timing tolerance is bounded by half a frame and the measured cadence jitter. An old pattern expires. A warning holds for one sample window and logs its event kind, period, cadence, timing tolerance and valid-pixel support. It reports delivery behavior, not a diagnosis of firmware, USB, Wi-Fi or hardware.

The saved clips have no ground-truth labels for periodic hardware dropouts. Uneven PC delivery alone must not be called a repeating fault. A synthetic periodic-depth fixture checks sensitivity separately from saved-recording warning counts.

## Sparse-camera tracking and timelapse

Large part registration retains its existing point minimum, fit, pose and marker guards. A previous measured bed remains a search hint when fresh dot fits fail; it cannot confirm a moved camera. Guidance masks do not remove points from tracking or saved frames. Timelapse rendering and recorded color synchronization are unchanged.
