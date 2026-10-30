# Changelog

All notable changes to PointYoink.

## 2.0.1

- One scanner panel: click the SCANNER tile and it has two tabs, Use and limits and Motion. View > Scanner motion opens the same panel on the Motion tab. The old floating Motion box and header button are gone.
- The motion stream connects by itself once the scanner is live over USB: it tries the address it remembers, then the neighbors and the rest of the Wi-Fi network, and checks that something there really sends motion data. If nothing answers it asks for the scanner's IP address once. The address is also remembered when the Wi-Fi transfer sees the scanner. Disconnect stops it from reconnecting.
- The Motion tab shows the scanner's tilt as a horizon with a heading notch, plus a ten-second trace of heading, pitch and roll. Two steps with pictures calibrate it (stand the scanner on its base, then lay it on its front face) so pitch, roll and heading follow how you actually hold the scanner; it is remembered between runs and Clear undoes it. Set heading to here calls the way it points now 0 without touching pitch and roll. The scanner window is larger so the Motion tab does not scroll on a big screen. The scanner graph is scaled to its real frame rate.
- Plain, Coverage and Distance now recolor the live surface right away, also while a scan is paused (they only changed the dots before).
- The scanner's fill light control is part of the app now, so the LED button works on a fresh install.
- Smaller: the repo's ignore list and two code comments no longer mention local-only files.

## 2.0.0

- Workspace home: one ribbon (Projects, Import, Scan, Build, Edit, Export) on every page, with Cut base, Smooth, Simplify and Line-up next to Build. Build, Edit and Export stay off until a project is open.
- Your projects: project cards with date, side count and size on disk, plus the total space all projects use. Show all past 20 projects. Open project from folder uses a real folder picker.
- Opening a project lands on Build when it has sides, on Scan when it has none. Reopening the project already open keeps its session.
- Saved projects open, build, edit and export with no scanner connected. The scanner status reads "Scanner not connected" until frames arrive.
- Import from scanner: whole projects over USB File Transfer or Wi-Fi Share to PC, through hidden staging with duplicate-safe names. Projects already on the PC are marked and asked about before importing again. Wi-Fi import no longer stops the camera.
- Export turns on as soon as a built model exists, reports on any page, and warns when the model on disk is older than the current settings.
- Scan: IR, Depth and Color feeds together, with a movable, resizable camera window. Scan and Build logs can be moved and resized.
- Usage readout at the top right: CPU, GPU and memory while scanning and building.
- Plainer words: "Coverage" for the where-to-scan-more stage, one name per tool (Line-up, Cut base), short hints for the fusion and mesh settings.
- Fixes: the YOLO Yoink sheet closes when you change pages, clicks that did nothing now say why, "1 side" instead of "1 sides".
- Scan (pre.13): Pause comes back at once, the voxel cache and live model are written in the background. A dragged live view keeps riding with the scanner from your angle; Home view snaps back. The turntable orbit lock now engages at the table's real turn rate in Small part mode too. The bed rim on the IR picture follows the newest dot fix and re-finds the table when the fixes disagree with it. The hint line says when the dots match none of the picked bed's, when the picked bed is not learned, and when the bed has left the picture. Learn a new bed asks how many dots it has and reports whether it found them; the map being built shows big while learning. Freehand (no bed) at the end of the bed list scans with no bed logic at all. The distance bars show the item in the middle of the picture, not the room behind it. The depth dots on the color picture stay with the picture they came with.
- Build and projects (pre.13): the Look row is two pair buttons (Shaded / Scan color, Flat facets / Points) with Overlay and View beside them, popups open upward, the info block shows the project's size on disk and the model file's size. The Before menu names dated versions. Project cards get a trash icon with a confirm. New project asks Save / Discard / Cancel. Camera setup is in Settings with the device steps spelled out; no launch flags. Menus get separators; Help has First scan steps; Preferences gets a General tab; Getting started rewritten; About links to GitHub.
