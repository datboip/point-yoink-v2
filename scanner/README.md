# PointYoink Scanner (the early command-line prototype)

Live scanning with the Revopoint MIRACO in PC mode on Linux (plain v4l2, no vendor SDK), tracked by
`track.py` in the repo root. This folder is the early prototype of the live flow, and its tracking and build code is what the 2.0 app uses. For everyday use run the app itself: `./venv/bin/python -m app2` from the repo root.

## No warranty, use at your own risk

This is experimental software from a hobby project. It is not made, approved or supported by Revopoint.
It talks to your scanner directly over USB while the scanner is in PC mode. Use it at your own risk: the
authors take no responsibility for damage to your scanner, your computer, your data or anything else.
It comes with no warranty of any kind, as set out in the LICENSE file in the repo root.

- It can hang or brick your scanner. If that happens, nobody here can fix it or help you recover it.
- Using it may void your scanner's warranty with the manufacturer.
- There is no support. Bug reports are welcome, but nothing is promised.
Stop the scan with Finish before unplugging the scanner, and do not leave it running unattended.

Run it from this folder with the repo's venv:

    ../venv/bin/python -u pointyoink_scanner.py            # new project
    ../venv/bin/python -u pointyoink_scanner.py --last     # continue the newest project

Projects are saved under `~/revopoint-scans-models/ProjectLive<stamp>/`, one folder per side.

## Files

| File | Job |
|---|---|
| `pointyoink_scanner.py` | The live window: sides list, big start/pause button, range and IR gain, Auto range, Auto gain (one gain per full turn), platter tint, Build, View in 3D |
| `table_plane.py` | Auto remove base: finds the turntable platter from the tracked frames of a side and saves `table_plane.json` |
| `bed.py` | Learn bed: the turntable's dot map, found in single frames (platter plane, centre, angle); `learn-sides` and `side-pose` for offline use |
| `support.py` | Platter support check: a side fit must keep both parts above both platters (throws out flipped fits) |
| `build_project.py` | Builds one model from all sides. Each heavy step is its own process under a memory cap |
| `strip_table.py` | Cuts the platter from one side's mesh |
| `keep_part.py` | Drops small loose pieces |
| `place_sides.py`, `refine_sides.py` | Consensus placement of the sides, then adjusting all of them together |
| `needs_scan.py` | Colours the built model by distance to open edges (red = scan here) |
| `render_views.py`, `closeup.py` | Offscreen preview renders |
| `rebuild_solid.py` | Poisson rebuild that keeps everything near real data (experiment) |
| `carve.py` | Removes what other sides looked through (experiment, NOT used by the build: it removed real surface) |
| `replay_track.py`, `track2.py` | Replay harness the tracker was developed with |

## What works (tested on a real part, 2026-09-26)

- Live tracking: sides of 200 to 440 frames with 0 to 2 lost frames.
- Platter found per side and cut in the build.
- 5 sides and 9 sides of the same part merged into one model, every side placed at 85 to 99 percent.
- Gain changes in the middle of a scan did not upset tracking.

## Learn bed

The platter's light dots are the brightest round blobs in the IR picture and their layout is not symmetric, so once the
dot map is known one frame places the whole platter. Press **Learn bed** once with the table empty and turning (one full
turn, about 40 s). The bed belongs to the turntable, not to a project: it is saved next to PointYoink's config
(`~/.config/pointyoink/bed.json`, or beside `$POINTYOINK_CONFIG`) and copied into each project that uses it. With no bed,
the first side scanned learns a provisional one from its own dots.

With a bed, every side finds the platter by frame 2-3 (3 agreeing frames, nothing under it) instead of after 8+ frames
and a turned part, and it can never take a flat face of the part for the table (the part has no dots). Auto remove base
still runs beside it: if the spin axis or Auto's plane disagrees the wrong way, or tracking and dots stop agreeing, Auto
takes over. A lost frame, or the first frame of a Continue after moving the scanner, is found again from the dots
(`Tracker.hint`) instead of the slow feature search. The build checks every side fit against both platters and throws
out fits that put a part inside a table; when the part stayed put between two sides the dots place it directly.
Offline: `bed.py learn-sides <side>... --out bed.json` learns a bed from saved sides.

## Known problems

- Adding more sides can make good areas worse. A thin lip seen from one face only is erased by sides that look past it.
- Thin edges and rims come out ragged. Clean-up and surface rebuilds do not fix that.
- `needs_scan.py` only flags open edges. It misses surface that is one skin thick and joined-but-rough edges.
- The live model preview is a rough point drawing, not the final surface.
- Changes made late on 2026-09-26 that have NOT run on the scanner yet: platter rim ring instead of filling
  the platter, Home button redraw while paused, scrolling sides list.
- The colour camera calibration and the default `--param` folder in `pointyoink_scanner.py` are for one scanner.

## Next steps, in order

1. Merge rule that keeps surface one side saw clearly unless another side saw something different there.
2. Track on the platter's marker dots and learn the empty table once (plane, size, dot map).
3. Build view that shows the sides coming together while the build runs.
4. Clean-up screen: presets, adjustable settings, before and after, manual delete. Edge tidy for thin parts.
5. Turn these scripts into modules with tests, then the six-screen app from the design doc.

## Rules that came out of testing

- Never call `np.linalg.svd` on an N x 3 point array without `full_matrices=False` (it builds an N x N matrix).
- Only one Open3D voxel grid may be alive in a process, and no `ray_cast` on a fresh grid.
- Every heavy mesh step runs under a memory cap.
- Let the user close the window with Finish so the scanner's stream stops cleanly.
