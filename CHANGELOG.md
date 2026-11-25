# Changelog

All notable changes to PointYoink.

## 2.0.6

- Update notice: once a day at start, PointYoink looks on GitHub for a newer release and says so once, with a Download button. Preferences > General can turn it off.
- YOLO Yoink has a Compare tab: your own part as each recipe built it, side by side, newest first, with Use this recipe on each card.
- Import: the progress bar shows only while copying, not while reading the scanner's project list.
- Scan page: the settings column in Advanced no longer runs 2 px past its panel (the right edge was clipped).
- A Revopoint RANGE plugged in is named as such and left alone (no light, gain or stream commands). RANGE support comes in 2.0.7.

## 2.0.5

- Fixed: the Points / Poisson build (the Revo-style recipes) cut every depth frame at 60 cm, even for Far scans and big parts, so anything farther was lost. It now uses each side's own depth limit, the same as the side build.
- Fixed: USB import could not connect while the desktop (GNOME) had opened the scanner by itself. The app now lets go of that connection and tries again, and the message no longer blames file manager windows.
- Fixed: Ctrl+C or a kill during an import left the app open. It now stops the transfer, releases USB and closes.
- Fixed: a Projects folder changed in Preferences did not reach the Import page.
- Import: clicking anywhere on a project row ticks it (Models), a second click clears it.
- The window fits smaller screens when it opens.
- The rebuild question no longer says "rebuilds 0 later edits".
- Third-party notices list every package the setup installs, with the tested versions.

## 2.0.4

- Help > Check for updates asks GitHub for the newest release when you choose it (nothing runs on its own).
- YOLO Yoink: every finished build and every Ultra Yoink step saves a picture of your own part under its recipe, and the cards and the big preview show it ("Your part, built ..."). A recipe with no picture yet shows a "No picture yet" tile. Card text is no longer cut off. The Advanced tab has Frames per side, Surface quality (Points only) and Smooth strength; Name and Save preset moved under the dropdown; Custom comes last. Ultra Yoink is a real button, hidden on Advanced.
- Ultra Yoink window: picture cards with readable text, a facts box on each card and one height for all cards.
- Simplify can keep the surface within a distance you set (Most the surface may move, in mm). It measures both ways, raises the kept share and tries again if the limit is missed, and reports "x% within y mm, farthest z mm".
- The What is left overlay says each thing once and draws the scan number beside each ghost scanner. Scanner spots and Frame all are off, with the reason in their tooltip, when the line-up is not saved. Turntable and Handheld have tooltips. The STL size estimate has its own line and is no longer cut off.
- View > Use this view as the project picture. The project picture is not saved while an overlay is on.
- Edit: Find loose pieces selects every floating piece (up to a size you set) so one Delete takes them. Grow and Shrink move by a distance in mm. The selection mode starts on New. Remove from an empty selection says so. Edit has its own Rebuild button when stages are out of date, and a rebuild that applies waiting deletes no longer leaves "1 delete waiting" behind. After a delete the faces hide at once, and the one rebuild starts by itself after four quiet seconds (not while faces are selected), or sooner with Rebuild. A delete no longer shows a loading card of its own; the rebuild after it does.
- When the scanner is on Wi-Fi but its USB side stopped answering, the app says "Scanner not on USB" and to turn the scanner off and on, instead of "connect the USB cable".
- Fixed: choosing a received project crashed with KeyError: 'mesh'. Own-part pictures of big models were skipped by a size limit that was too low. The memory limit no longer stops the Simplify distance check on large scans.

## 2.0.3

- Fixed: `process.py` and `viewer.py` were missing from the repo and from every package so far, so Simplify, Fill holes, Slivers and the Ball pivoting mesh method did not work in 2.0.0 to 2.0.2. They are in now, and tests check that every script the engine runs is present and that Simplify runs end to end. If you use those tools, install this version.
- While "What is left" or the quality check runs, only the progress card in the view shows it. The second line and bar in the side panel and the top banner are gone.
- The scan guide's two modes are called Turntable and Handheld, with a "Next scan" label. The ghost scanners button is "Scanner spots", Frame everything is "Frame all", and the orange "can't suggest" messages say what is missing in plain words.
- README and Getting started corrected: the real ribbon (Projects, Import, Scan, Build, Edit, Export, with Build, Edit and Export off until a project is open), where the resource tiles are, the Export sheet, the Edit tools, where files and settings go, the first-launch download, and the SCANNER window. The internal design notes in docs/ are gone from the repo.

## 2.0.2

- The 2.0.1 package left out three files, so it would not start. Fixed: the package is rebuilt, and the build now stops when source files are not committed.
- The setup window says which graphics card it found ("GPU detected: ...") or that there is none, and has a Change folder button for choosing where the downloaded parts go. The choice is remembered.
- The motion stream only looks for the scanner on home and office network addresses, and only remembers an address of that kind.

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
