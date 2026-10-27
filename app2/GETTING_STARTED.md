# Your first model

**Scan → Build → Edit → Export.** You can go back to any step. No scanner yet? **Import** a project scanned on the MIRACO and build it here.

## 1. Set up the camera, once

The MIRACO is calibrated at the factory. The app only needs a copy of its camera files, once per camera.

1. On the scanner itself, not PC mode, scan anything with **Small part** and save it.
2. Do the same with **Large part**.
3. Plug in USB, pick **File Transfer** on the scanner, then **File → Import scanner projects** and import both.

**Settings → Camera setup** on the Scan page shows which camera is set up.

## 2. Open or name a project

**Projects** in the top bar lists your projects with a picture, side count and size. **Open** opens one. The trash icon at the card's left edge deletes it, after asking.

**File → New scan** starts a new one. With a project open it asks whether to save it, discard it (the folder is deleted) or cancel. Name the new one in the header and press Enter. The name is for the app, the folder keeps its scan number.

## 3. Scan the part

**Help → First scan steps** walks these steps one at a time on the Scan page.

1. Plug in USB and pick **PC mode** on the scanner.
2. **Learn bed.** With table dots, take the part off and let the empty table make a full turn. **Learn a new bed...** in the Bed card's list asks the bed's name and how many dots it has, so the result can say whether it found them all. The last entry, **Freehand (no bed)**, scans with no bed at all: nothing is cut and the part or the scanner can move freely.
3. **Place the part** in the middle of the bed. Hold the scanner so the distance bar reads **Best**. Keep **Range** and **IR gain** on Auto to start.
4. **Start side.** Turn the part or move the scanner slowly. **Pause** stops adding frames and saves in the background, so **Continue** is there at once. **Back 30 / 10 / 5 / 1** drops the last frames if tracking went wrong.
5. **Flip the part** and click **+ New side.** Keep some of the same shape in view so the sides can line up.
6. **Build combined model** when the sides cover the part.

**Touch up** scans a built model again only where it needs it: red is a hole, purple is rough. Aim at them until they turn green, or at any area you want better.

**Fill LED** is the scanner's light. After five minutes the app asks whether you still need it.

## 4. Build a solid model

**Build combined model** uses the current settings. **YOLO Yoink** picks a recipe first: **Classic** for everyday, **Sharp** for fine detail from many sides, **Quick Peek** for a fast look, **Detail** or **Deep Clean** for a smooth new surface, **Print** for a small closed model.

The stages: **Side** builds each view and cuts the table under it, **Line-up** fits the views together, **Fused** joins them, **Meshed** makes the surface, **Coverage** shows in red where to scan more.

Each tool tab shows what it will change before **Apply**: **Fill holes** outlines the holes it closes in blue and the ones it leaves in red. **Detect** tools show the affected area in red first.

**Compare** shows the model side by side with an earlier one. The **Before** button says which: the last tool, all tools, or a dated version.

## 5. Look at the model

Along the bottom of the view: **Look** has two buttons. **Shaded** flips to **Scan color**, **Flat facets** flips to **Points**. **Overlay** colors the model by **What is left** to scan, **Frame support** or **Capture distance**. **Home view** and **View** sit above the corner cube. Click a face of the cube to look straight at that side. While scanning, a dragged view keeps riding along with the scanner from your angle; Home view snaps back to the scanner's own view.

**View → Perspective** or **Orthographic**. **Set this view as Front** or **Top** saves the direction with the project without rotating the mesh.

## 6. Edit the shape

**Edit** in the top bar. **Rectangle**, **Lasso** and **Brush** select faces. **Grow**, **Shrink**, **Connected piece** and **Magic** widen a selection. **Through** also takes what is hidden behind. **Delete** removes the selection, **Undo** brings it back.

Shape tools pick a spot on the surface, then show red **X, Y, Z arrows** on it, like a slicer. Drag an arrow to slide the spot along one axis, Shift for whole millimeters.

- **Cut shape** removes a cylinder, box or sphere. Click again to move it.
- **Add fin** needs a closed mesh: two spots set its ends, a third which way it faces.
- **Cut gap** removes a narrow slot between two spots.
- **Smooth area** and **Fill gap** work on one spot.

Red is always what Apply will change. Inspect it first.

## 7. Save for printing

**Finish for print** in Build sets the triangle count, smoothness and hole filling. **File → Export STL** saves it, **Export...** offers OBJ, PLY and 3MF. Check scale, thin parts and intended holes in your slicer.

**File → Export scan timelapse...** makes a GIF or MP4 of a side growing.

## 8. Import from the scanner

**Import** in the top bar. **USB File Transfer** or **Wi-Fi Share to PC**. Tick projects, then **Import selected projects**. Already imported ones say so. **Done** releases the scanner, then switch it back to PC mode.

**Screenshots** and **Screen recordings** tabs copy the scanner's own captures to the folder set in **Preferences**.

## Find help

The **?** buttons explain nearby tools. **Help → Keyboard shortcuts** lists the keys. **Help → Report a problem...** gathers the log for a report.
