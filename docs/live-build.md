# Live build

Choose **Live build** under Capture. The accepted views fuse into the model as
you scan. The counters show frames seen and frames kept. A repeat can improve
the live average without adding another saved frame.

In **Live settings**, choose Fine, Balanced, or Coarse detail. The sizes come
from measured depth noise and pixel spacing for the selected camera mode.
Fine uses more memory. Coarse can erase thin walls. The default target is the
tracker's existing 12 good views per region. The saved rule also records the
minimum share of surface a frame must improve, initially 3% as in Smart frames.
A frame helps by adding surface, improving its best distance band, or adding
good-distance support below the target. This is a saving rule, not proof that
an unseen surface has been scanned.

The model's Frame support view shows unfinished regions. The turn ring fills
only from kept frames; the moving tick uses the current bed angle even while
paused. A pass can finish only after its observed regions reach the support
target and its kept turn slices are covered. Hidden surfaces still need a new
angle. Low suggests about 25 degrees down, Level looks from the side, and
Flipped exposes the bottom. Pause and choose **Next pass** to make another
side with its own ring. The Build pipeline joins sides as usual.

Lost tracking pauses capture. **Undo recent** pauses and restores the exact
recent voxel snapshot, including contributions from repeats that were never
saved. Its history is bounded by the existing six-checkpoint limit and the
recent three-second window, so at three frames per second it normally covers
about two seconds. It does not undo an entire long pass. Resume only after
checking the model. Pose drift can still be baked into a live surface.

Pause to save or export the local live mesh. Reopening validates and restores
the exact local voxel cache. Changed depth, poses, calibration, detail, or
implementation invalidate it. Moving a side in the assembled model does not
resample its volume. If the cache is missing or invalid, the side rebuilds
from kept frames, so unsaved repeat averaging cannot be recovered.

**Build combined model** remains available. It runs the full quality pipeline
from kept frames only. The live mesh is an early result, not the same output
as the combined cleanup and meshing pipeline. Keep Continuous and Smart frames
available: saved replay does not yet establish that Live should replace them.

## Surface and light

Normal, Dark, Shiny, and Sprayed set starting settings. They do not change
settings again during capture. Near starts were chosen from observed fill,
depth noise, and clipping at saved gains. Angles and materials varied between
those views, so the presets need controlled device testing. Far has only
gain 1 evidence and intentionally offers no unmeasured gain sweep. Projector
registers are left unchanged.

The surface hint describes missing depth with low IR, saturated IR, or a good
return. It does not identify the material. Try another angle or spray when
appropriate. Color-only missing-depth marks require foreground contrast and
steady evidence. Same-color parts and backgrounds can remain unknown.

**Light during scans** lets the app own a light on Start and turn it off on
Pause or Stop. A manually enabled light stays under manual control.

## Range test and camera marks

Use an empty learned mat for the four-step Range test. Sweep slowly, holding
each distance. A short result stays partial. Enough repeated distance bins
must cover at least half the shipped span. Nothing replaces the bands until
you click **Apply**. The distance bar identifies measured bands and offers
**Use default bands**.

Show marks offers grid density, circle size, and a shared distance palette.
Blue, orange, and magenta stay distinct in the documented color-vision
simulation, but the gray Range cut can resemble magenta under some simulated
conditions. Read the key and Range values as well as the colors. Random dots
are limited to the Range test; normal capture uses the aligned grid.

The dashed line shows only the independently observed stereo overlap edge.
Where a mat or background ends before that optical limit, the line stays
unknown. It never wraps the part or fills a hull across empty space.
Optional one-dot and adjacent two-dot rim markers provide a fallback angle
when learned top dots are hidden. They need valid rim depth and a successful
Learn bed phase and height measurement. Without those markers the top-dot
map continues to work. No hardware marker behavior is established by the
synthetic regression.
