# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's words: every label, control, range, default, factory value, hint, button text, progress line, notice
and key of the page, as data. The panels are built from it, so the wording lives in one place, and a test
(qtb/tests/test_words.py) checks it against the engine: every `path` resolves with stages.params, every range and
default equals the engine's Param, and no string has an em or en dash.

Wording: our own plain words, sentence case, American spelling, units as "(mm)" after a space. Tab `name`s are the
keys the page, the ribbon and saved state use; a tab's `label` (when set) is the text it shows.

Numbers: `default` is OUR default (the engine's Param.default, what a fresh project gets); `revo` is the scanner's
factory default, drawn as a small tick on the slider track when it differs. Ranges and steps are the engine's. A control's
live value comes from the recipe (stages.params) at run time; the numbers here are the reference."""

# ------------------------------------------------------------------------------------------------ the stage list (left)
# id = graph.GROUPS name; row = the big label; panel = the title of the tool panel it opens on the right (tapping
# Fused shows the "Fusion" panel);
# report is "Coverage" (where the scan is thin); "Scan here" is only its red legend item ("scan this area more");
# sub = the small second line, filled from status(); None = no second line.
STAGES = [
    dict(id="raw", row="Raw frames", panel="Raw frames", sub="{sides} sides, {frames} frames"),
    dict(id="side", row="Side", panel="Side", sub="Per side, before the line-up", group="side"),
    # the Side stage's steps, one row each (they open Side's panel on that tab)
    dict(id="base", row="Cut base", panel="Side", sub="Removed on {cut} of {sides}", indent=True, tab="Platter", stage="side", in_group="side"),
    dict(id="isolation", row="Loose pieces", panel="Side", sub="Drops small floating bits", indent=True, tab="Isolation", stage="side", in_group="side"),
    dict(id="sidefusion", row="Side detail", panel="Side", sub="Point spacing", indent=True, tab="Fusion", stage="side", in_group="side"),
    dict(id="lineup", row="Line-up", panel="Line-up", sub="{active} sides in, base side {base}"),
    dict(id="fused", row="Fused", panel="Blend", sub="{method} {pd} mm, {ntools} tool{s}", group="fused"),
    dict(id="fz_fusion", row="Blend", panel="Blend", sub="", indent=True, tab="Fusion", stage="fused", in_group="fused"),
    dict(id="fz_isolation", row="Loose pieces", panel="Blend", sub="", indent=True, tab="Isolation", stage="fused", in_group="fused"),
    dict(id="fz_cut", row="Cut base", panel="Blend", sub="", indent=True, tab="Cut base", stage="fused", in_group="fused"),
    dict(id="fz_overlap", row="Overlaps", panel="Blend", sub="", indent=True, tab="Overlap Detection", stage="fused", in_group="fused"),
    dict(id="fz_smooth", row="Smooth", panel="Blend", sub="", indent=True, tab="Smooth", stage="fused", in_group="fused"),
    dict(id="meshed", row="Meshed", panel="Mesh", sub="{mesh_method}, {ntools} tool{s}", group="meshed"),
    dict(id="ms_mesh", row="Mesh", panel="Mesh", sub="", indent=True, tab="Mesh", stage="meshed", in_group="meshed"),
    dict(id="ms_isolation", row="Loose pieces", panel="Mesh", sub="", indent=True, tab="Isolation", stage="meshed", in_group="meshed"),
    dict(id="ms_cut", row="Cut base", panel="Mesh", sub="", indent=True, tab="Cut base", stage="meshed", in_group="meshed"),
    dict(id="ms_simplify", row="Simplify", panel="Mesh", sub="", indent=True, tab="Simplify", stage="meshed", in_group="meshed"),
    dict(id="ms_smooth", row="Smooth", panel="Mesh", sub="", indent=True, tab="Smooth", stage="meshed", in_group="meshed"),
    dict(id="ms_cleanup", row="Clean up", panel="Mesh", sub="Click scraps off, fix spots (opens Edit)", indent=True, stage="meshed", in_group="meshed", edit=True),   # 2026-10-25
    dict(id="report", row="What is left", panel="What is left", sub="{places} place{s} to scan"),
    dict(id="orient", row="Orient", panel="Mesh", sub="Top and front, for export"),      # a shortcut: the Mesh panel's Orient block
    dict(id="textured", row="Textured", panel=None, sub="Color comes later", disabled=True),
]

# a row's state, from NodeInfo.status of the stage's last node (and retry / adopted); text is right-aligned in the row
ROW_STATE = {
    "built_retry": dict(dot="warn", text="Built, will retry", tip="{warn}"),
    "fresh":   dict(dot=None,   text=""),
    "adopted": dict(dot=None,   text="", tip="Taken from the earlier one-shot build"),
    "stale":   dict(dot="warn", text="Will rebuild", tip="Press Rebuild the rest, or Apply here, to update it. A step before it changed."),
    "waiting": dict(dot="ink3", text="Waiting", tip="Builds after the stage above it."),
    "new":     dict(dot="ink3", text="Waiting", tip="Not built yet. Press Build it to build every step."),
    "running": dict(dot="accent", text="{pct}%"),
    "failed":  dict(dot="lost", text="Failed", tip="{error}"),
    "retry":   dict(dot="warn", text="Will retry", tip="{warn}"),
    "gone":    dict(dot="warn", text="Already gone", tip="Nothing left to delete here: an earlier delete already took this area."),
    "check":   dict(dot="warn", text="Check", tip="Look this step over: a delete removed {now} instead of {was}."),
    "n/a":     dict(dot=None,   text=""),
}

# the info block above the stage list; counts are printed without separators ("Points: 1860966")
INFO = {
    "line1": "{sides} sides, {frames} frames",
    "line2": "Camera: {module}",               # "Near" from the project's property.rvproj; hidden when unknown
    "raw": "Frames: {frames}",
    "side": "Side {n} of {sides}    Faces: {faces}",
    "side_head": "Side {n} of {sides}",                  # the face count is not known yet
    "lineup": "Sides in: {active} of {sides}    Agree: {agree} mm",
    "lineup_head": "Sides in: {active} of {sides}",       # the agreement is not known yet (no "Agree: ? mm")
    "points": "Points: {points}",
    "mesh": "Vertices: {verts}    Polygons: {faces}",
    "faces": "Polygons: {faces}",                       # the vertex count is not known yet
    "report": "Open edge: {edge_mm} mm in {places} places",
    "size": "Size: {dims}",                             # the finished model's box, "W x D x H mm" (Meshed stage)
    "project": "Project: {size}",                       # the project folder on disk, "21.4 GB"
    "file": "File: {size}",                             # the model file on screen, "34.4 MB"
    "stl": "STL about {size}",                          # what an STL export would weigh
}

# ------------------------------------------------------------------------------------------------ control helpers
def slider(label, path=None, tool=None, param=None, lo=None, hi=None, step=None, default=None, revo=None, unit="",
           fmt="{:g}", hint="", advanced=False, show_if=None):
    """path = a recipe param path (e.set); tool+param = a tool's param (e.add_op / e.set_op)."""
    return dict(kind="slider", label=label, path=path, tool=tool, param=param, lo=lo, hi=hi, step=step, default=default,
                revo=revo, unit=unit, fmt=fmt, hint=hint, advanced=advanced, show_if=show_if)


def seg(label, path, choices, default, hint="", advanced=False, show_if=None):
    """choices = [(engine value, chip text, one-line hint under the chips when picked)]"""
    return dict(kind="seg", label=label, path=path, choices=choices, default=default, hint=hint, advanced=advanced, show_if=show_if)


def toggle(label, path, default, revo=None, hint="", advanced=False, show_if=None):
    return dict(kind="toggle", label=label, path=path, default=default, revo=revo, hint=hint, advanced=advanced, show_if=show_if)


RESET_TIP = "Back to the applied value, then to {default} (default)"                 # the circular arrow after every slider label
RESET_TIP_REVO = "Back to {default} (preset default)"     # a tool whose new step starts at the factory value
REVO_TICK_TIP = "Preset default: {revo}"                 # the small tick under the track at the factory value
OURS_TICK_TIP = "Classic Yoink uses {revo}"               # the tick when the slider starts at the factory value (a new tool step)
MORE = "More settings"                                   # disclosure for advanced=True controls; closed by default
APPLY, DETECT, DELETE, CANCEL, UPDATE = "Apply", "Detect", "Delete", "Cancel", "Update"
APPLY_ALL = ("Apply to all sides", "Off: these settings are for the side shown. On: every side gets them.")   # Side panel switch

# ------------------------------------------------------------------------------------------------ the panels (right)
# Each panel: tabs; each tab: controls and one full-width button (Apply changes the data now, Detect first shows in
# red what would go, then Delete / Cancel). run_to = what Apply runs (never past its stage).
# Cut base: what the app found on this side, two big choices, a Keep switch, and the small Where to look chips under Cut it for me.
CUT = dict(
    forme=("Auto", "Finds the base on its own: the board dots, the turntable seen while scanning, or a search of the side. The line above says which."),
    hand=("Manual", "Click 3 or more spots on the flat base in the view, spread apart. The violet shows what goes. Then press Remove base."),
    keep=("Keep the base", "Nothing is cut on this side. For a part lying flat that the cut would eat into."),
    where="Where to look",
    where_choices=[("auto", "Turntable", "The turntable seen while scanning, else a search of the side."),
                   ("plate", "Plate", "The top of a plate or riser the part sits on, up to 15 cm above the turntable. Turntable when unsure."),
                   ("search", "Search", "Ignores the plane saved while scanning and searches the side again.")],
    found={"board dots": "Base found by the board dots.", "live plane": "Base found: the turntable seen while scanning.",
           "plate": "Base found: the top of the plate.", "auto": "Base found by searching the side.", "search": "Base found by searching the side.",
           "manual": "Base cut by hand."},
    crossed="Base still showing? Use Manual. Auto left this side uncut: the only base it found would cut into the part.",
    failed="Mark the base by hand (Manual), or build the side again. The base cut failed, so nothing was cut.",
    kept="Base kept: nothing is cut on this side.",
    unbuilt="Not built yet: the base is looked for when the side builds.",
    all_sides="Turn off Apply to all sides before marking by hand: the spots belong to one side.",
    share=" Cutting {pct}% of this side.",
    share_big=" Cutting {pct}% of this side: check that the violet is only the base.",
    hand_wait="Base not marked yet: click 3 spots on it in the view.", hand_ready="Base marked by your spots.",
)

HOLES = dict(                                  # Fill holes: the outlines drawn before Apply
    count="Closes {n} of {m} holes at {mm} mm{auto}. Blue outlines close; red stay open.",
    left="The biggest left open is about {big} mm across: raise Largest hole to close it.",
    closed="No holes: the surface is closed already.",
    no_mesh="Build a mesh first: Fill holes works on a mesh.")
SPOTS = dict(
    info="Click 3 or more spots on the flat base the part sits on, spread apart. The violet shows what will be removed. Then press Remove base.",
    count="{n} spot{s} picked", count_need="{n} of 3 spots picked",
    undo="Undo spot", clear="Clear", setbase="Remove base", prev="\u2039 Previous side", nextb="Next side \u203a", allh="Use this plate height for all sides",
    allh_tip="Sets Margin to this plate's thickness above the turntable and applies it to every side (Auto finds the turntable on each).",
    allh_done="Plate is about {mm} mm thick: Margin set to {m} mm on all sides.", allh_bad="Could not compare with this side's turntable plane. Build the side first.",
    need="Click at least 3 spots on the base first, spread apart (not in a line).",
    done="Removing the base on side {side} (cut through {n} spots).",
    flat="Those spots are almost in a line. Pick some further apart.",
    cut_info="Click 3 or more spots on the flat base, spread apart. The red shows what goes: everything below the spots, inside the area they outline. Then press Remove base. If some is left, click again and press it again: each press is its own layer in the list.",
    cut_done="Base removed ({n} spots). Layer {k} is in the list.", cut_big="Check the violet before you press: this cut removes {pct}% of the model.",
    missed="Click on the model itself.",
    clone_info="Click the spot to copy from, then the spot to paste onto. Green is what is copied, red is what it replaces.",
    mirror_info="Pick 3 spots along the middle, or press Auto-find. Purple shows the mirror plane. Red shows gaps the copy would fill. Measured surfaces stay.",
    mirror_auto="Auto-find the plane",
    mirror_go="Apply mirror",
    mirror_need="Pick the mirror plane first: 3 spots down the middle, or Auto-find.",
    mirror_state="Halves agree {pct}%. The mirror would add {n} points or faces.",
    mirror_low="Check the red before applying: only {pct}% of the part matches its mirror image. The part may not be symmetric here, or the plane is off.",
    mirror_plane="Mirror plane",
    mirror_done="Mirrored: the other side was filled in from this one. It is a guess made from the scan, listed as its own faces.",
    fix_go="Apply smoothing",
    fill_go="Fill the gap",
    fill_info="Click beside the gap and size the area to cover it. The new surface is a guess that continues the surface around it: it is kept as its own faces, and the outline shows where it goes. Works on the finished model only.",
    fill_done="Gap filled with a guess. Undo or turn the edit off to take it away.",
    fix_info="Click the rough area. Only the shape around it is smoothed, fading out at its edge. Red shows what changes.",
    fix_done="Smoothed the area. It is a layer in the Applied list.",
    fix_need="Click the area to smooth first.",
    clone_need="Set the source and the destination spots first.",
    clone_done="Cloned: the copy replaces {n} points or faces. It is made-up geometry copied from the scan.",
    cut_part="This cut reaches deep into the model. Are the spots on the base, not on the part?",
    hand_done="base cut by hand", building="Side {side} is not built yet: building it so you can click the base.", all_done="Every side in the model has a by-hand base cut.",
)

ORIENT = dict(
    spots="Bottom: click 3 or more spots on the face that should lie on the bed, spread apart.",
    count="{n} spot{s} picked", undo="Undo spot", clear="Clear",
    setbottom="Set bottom from spots", front="Use this view as Front", reset="Reset orientation",
    state_set="Orientation set. Export uses it.", state_none="Not set. Export keeps the scan's own orientation.",
    need="Click at least 3 spots on the bottom face first, spread apart (not in a line).",
    flat="Those spots are almost in a line. Pick some further apart.",
    nomodel="Build the model first, then orient it.",
    saved_bottom="Bottom set from {n} spots.", saved_front="Front set to this view.", cleared="Orientation reset.",
)

PANELS = {
    "raw": dict(title="Raw frames", tabs=[dict(
        name=None, controls=[], button=None,
        rows="Side {n}    {when}    {frames} frames{lost}",      # lost = "    12 not tracked" when the scanner lost some frames, else ""
        apart_many="Sides {ns} were scanned {gap} from the rest. Same part? Nothing to do. A different part? Untick Use in Line-up for them.",
        apart="Side {n} was scanned {gap} from the rest. Same part? Nothing to do. A different part? Untick Use in Line-up for it.",
        extra=[("Resume scan", "Continue this side on the Scan step"),
               ("Rebuild this side", "Build this side again from its saved frames")],
        delete_tip="Tick the sides that came from the wrong scan, then delete them. This can't be undone.",
        delete_ask="Delete {what}? Their frames are removed from this project. This can't be undone.",
        hint="Frames as the scanner saved them. Resume scan adds frames to this side.")]),

    "side": dict(title="Side", tabs=[
        # name = the key the page and the Workspace ribbon use; label = the tab's text (one word per concept: "Cut base")
        dict(name="Platter", label="Cut base", controls=[
            dict(kind="cutchoice", label="Cut base", path="side[{id}].platter.method", default="auto"),   # CUT below: the words
            slider("Margin (mm)", "side.platter.margin", lo=0, hi=40, step=0.1, default=1.0, unit="mm",
                   hint="Also cuts this much above the turntable, on every side. Raise it if bits of turntable are left; lower it if the part's bottom is cut."),
            slider("Max tilt (deg)", "side.platter.max_tilt", lo=1, hi=20, step=1, default=8, unit="deg",
                   hint="How far the turntable may lean and still count as the base, on every side. Raise it if a tilted base is not found.", advanced=True)],
             apply_to=True, button=APPLY, run_to="side.keep@{id}"),
        dict(name="Isolation", label="Loose pieces", controls=[
            slider("Loose piece size", "side.keep.min_share", lo=0, hi=100, step=1, default=3, revo=15, unit="%",
                   hint="Drops floating bits smaller than this share of the biggest piece. Raise it if junk stays; lower it if real small parts go.")],
             apply_to=True, button=DETECT, run_to="side.keep@{id}"),
        dict(name="Fusion", label="Detail", controls=[
            slider("Point spacing (mm)", "side.fuse.voxel", lo=0.2, hi=2.0, step=0.05, default=0.4, revo=0.4, unit="mm",
                   hint="Only used to line the sides up; the model's own spacing is on Fused. Lower it if sides line up badly."),
            slider("Blend width (cells)", "side.fuse.trunc", lo=1, hi=8, step=0.5, default=3, advanced=True,
                   hint="How far each surface reaches when frames blend. Raise it to merge doubled walls; lower it to keep thin sheets."),
            slider("Min. views", "side.fuse.min_weight", lo=1, hi=20, step=1, default=3, advanced=True,
                   hint="Drops spots seen in fewer frames than this. Raise it to drop noise; lower it if thin parts vanish."),
            slider("Edge filter (mm)", "side.fuse.filter", lo=0, hi=10, step=0.5, default=3, advanced=True,
                   hint="Drops depth at sudden jumps bigger than this, often noise at edges. Raise it if edges look fuzzy. 0 = off."),
            slider("Edge-on limit (deg)", "side.fuse.grazing", lo=45, hi=89, step=1, default=80, advanced=True,
                   hint="Drops spots the camera saw more edge-on than this angle. Lower it if side walls look noisy."),
            slider("Max depth (mm)", "side.fuse.max_depth", lo=200, hi=1000, step=10, default=600, advanced=True,
                   hint="Ignores anything farther from the scanner than this. Lower it if the background shows up."),
            slider("Use every Nth frame", "side.fuse.every", lo=1, hi=10, step=1, default=1, advanced=True,
                   hint="Uses one frame in N. Raise it for a faster, rougher build."),
            toggle("Use the graphics card", "side.fuse.gpu", True, advanced=True,
                   hint="Builds faster on the graphics card. Turn off if building crashes or runs out of graphics memory.")],
             apply_to=True, button=APPLY, run_to="side.keep@{id}")],
        delete_note="To change the model, delete on Fused or Meshed. Deleting on a side only helps the sides line up."),

    "lineup": dict(title="Line-up", tabs=[
        dict(name="Sides", controls=[], button=APPLY, run_to="lineup.adjust",
             head="How well each side lands on the others",
             row="Side {n}    {score}%",               # score = placement score share; colour kept >= 80, warn 50-79, lost < 50
             row_base="Base, stays put",
             row_use="Use",                           # checkbox: leave out / include
             row_eye="Show",                          # view only, no engine change
             weak="Weak fit: check it lines up, or flip it",
             agree="Sides agree to {after} mm (before refining: {before} mm)",
             apply_n="Apply {n} change{s}", apply_none="Apply"),
        dict(name="Flip", controls=[
            seg("Axis", None, [("auto", "Auto", "Tries each flip and keeps the best fit."),
                               ("x", "X", "About the side's longest axis."), ("y", "Y", "About its middle axis."),
                               ("z", "Z", "About its shortest axis.")], "auto")],
             side_pick="Side", button=APPLY, run_to="lineup.adjust",
             hint="Turns the side 180 degrees about one of its own axes and fits it where it overlaps the other sides. "
                  "It keeps that spot when the line-up runs again.",
             preview_wait="Fitting the flip...",
             list_head="Flips and point matches", list_row="Side {n}, {axis} axis", list_remove="Remove"),
        dict(name="Points", controls=[], button="Line up from points", run_to="lineup.adjust",
             side_pick="Move side", to_pick="Match to",
             hint="Click a spot on one side, then the same spot on the other. 3 pairs minimum, 5 spread out is better.",
             next_start="Next click: pair {n}, on either side", next_other="Next click: the same spot on side {side}",
             next_redo="Next click: pair {n} again, on side {side}", wrong_side="Pair {n} is being redone on side {side}: click there.",
             pair_row="Pair {n}", redo="Side {side}", redo_tip="Click this pair's spot on side {side} again",
             delete="Delete", undo="Undo point", clear="Start over",
             need="Pick {k} more pair{s} to line up.", same="Pick two different sides.",
             collinear="The points lie on one line: spread them out.", duplicate="Two points are in the same spot.",
             fit="Fit to the points: {mm} mm on average.", list_head="Flips and point matches"),
        dict(name="Nudge", controls=[
            dict(kind="xyz", label="Move (mm)", key="t", step=0.1, big=1.0, lo=-50, hi=50)],
             side_pick="Side", button=APPLY, secondary="Reset", run_to="lineup.adjust",
             rings="Drag a ring in the view to turn the side about X (red), Y (green) or Z (blue). Drag on or outside the ring to turn freely, inward over the ticks for 5 degree steps, "
                   "further in by the spokes for 30 degree steps. Shift: at least 15 degree steps.",
             rel_head="Rotate (relative, deg)", rel_tip="Type an angle and press Enter to turn the side that much about this axis",
             abs_head="Rotate (absolute, deg)", abs_tip="The turn from where the side sits now, about X, then Y, then Z",
             turned="Turned {deg} degrees so far", not_turned="Not turned",
             keep="Keep exactly as placed", keep_tip="Skip the fit: the side stays exactly where you put it",
             hint="A move is applied after refining, so refining never undoes it. A turn is fitted to the other sides "
                  "from where you put it, and stays put if the fit would move it more than a few mm.",
             list_head="Nudges and turns", list_row="Side {n}: {t} mm, {r} deg", list_remove="Remove",
             turn_row="Side {n}: turned {deg} deg", turn_row_keep="Side {n}: turned {deg} deg, kept as placed"),
        dict(name="Settings", controls=[
            dict(kind="combo", label="Base side", path="lineup.place.base",
                 choices=[("auto", "Auto (fits best with the others)"), ("{id}", "Side {n}")], default="auto"),
            slider("Pair fit spacing (mm)", "lineup.pairs.voxel", lo=0.3, hi=5, step=0.1, default=1.0,
                   hint="How finely side pairs are compared. Lower it for a closer fit; raise it if lining up is slow."),
            slider("Landing tolerance (mm)", "lineup.place.tol", lo=0.3, hi=5, step=0.1, default=1.5,
                   hint="A point counts as landing on another side when this close. Raise it if rough sides will not line up."),
            seg("Refine", "lineup.refine.method", [("pose_graph", "All together", "Fits all sides together after placing. Best for most scans."),
                                                   ("none", "Off", "Keeps each side where it was placed. Try it if refining pulls a side out of place.")], "pose_graph"),
            slider("Refine spacing (mm)", "lineup.refine.voxel", lo=0.2, hi=3, step=0.1, default=0.5, advanced=True,
                   hint="How finely the last fit compares sides. Lower is closer but slower."),
            slider("Refine rounds", "lineup.refine.rounds", lo=1, hi=6, step=1, default=2, advanced=True,
                   hint="More rounds can fit the sides closer, but take longer."),
            slider("Samples per side", "lineup.pairs.samples", lo=20000, hi=1000000, step=10000, default=250000, advanced=True,
                   hint="Points taken from each side to fit it. More is closer but slower."),
            slider("Pair fit reach (mm)", "lineup.pairs.max_dist", lo=1, hi=20, step=0.5, default=6.0, advanced=True,
                   hint="How far apart two sides may start and still be fitted. Raise it if sides start far off.")],
             button=APPLY, run_to="lineup.adjust",
             extra=[("Refit all pairs", "Fit every pair of sides again from scratch. Try it if sides land in the wrong place.")])]),

    # the Fusion panel: four tabs
    "fused": dict(title="Blend", list_token="fused", tabs=[
        dict(name="Fusion", label="Blend", controls=[
            seg("Blend method", "fused.method",
                [("tsdf", "Standard", "Blends every frame into one surface. Fast, good for most parts."),
                 ("points", "Points", "Keeps the scanner's points and builds the surface after. Slower, sharper edges."),
                 ("*", "{Method.label}", "{Method.help}")], "tsdf",     # any other registered method (plus) gets a chip
                hint="") | dict(extra_words={                             # plain words for those chips (the engine says "TSDF plus")
                    "plus": ("Plus", "A sharper surface, and drops surface sealed inside the part. Slower.")}),
            seg("Scan color", "fused.color", [("auto", "Auto", "Uses color when every side in the model has it. Adds about 10 to 15 seconds per side."), ("on", "On", "Uses the saved color frames. Adds about 10 to 15 seconds per side."), ("off", "Off", "Builds shape only, without scan color. Faster.")], "auto", hint="Adds about 10 to 15 seconds per side. Turn off when you only print in one color."),
            toggle("Auto white balance", "fused.white_balance", True, hint="Makes a gray part look gray under warm or cool light. Turn off for parts meant to look warm or cool."),
            slider("Point spacing (mm)", "fused.point_distance", lo=0.1, hi=2.0, step=0.05, default=0.4, revo=0.4, unit="mm",
                   hint="Smaller keeps more detail but is slower and needs more memory. Raise it for a quick look.",
                   ),
            slider("Frames per side", "fused.frames_per_side", lo=0, hi=5000, step=50, default=300, advanced=True,
                   fmt="{:g}", hint="Uses at most this many frames a side, evenly spread. Raise it if detail is missing. 0 = all."),
            # method params go under More settings, from e.schema("fused", method), with these labels:
            slider("Blend width (cells)", "fused.tsdf.trunc", lo=1, hi=8, step=0.5, default=3, advanced=True, show_if="tsdf",
                   hint="How far each surface reaches when frames blend. Raise it to merge doubled walls; lower it to keep thin sheets."),
            slider("Min. views", "fused.tsdf.min_weight", lo=1, hi=20, step=1, default=3, advanced=True, show_if="tsdf",
                   hint="Drops spots seen in fewer frames than this. Raise it to drop noise; lower it if thin parts vanish."),
            slider("Edge filter (mm)", "fused.tsdf.filter", lo=0, hi=10, step=0.5, default=0, advanced=True, show_if="tsdf",
                   hint="Drops depth at sudden jumps bigger than this, often noise at edges. Raise it if edges look fuzzy. 0 = off."),
            slider("Edge-on limit (deg)", "fused.tsdf.grazing", lo=45, hi=89, step=1, default=80, advanced=True, show_if="tsdf",
                   hint="Drops spots the camera saw more edge-on than this angle. Lower it if side walls look noisy."),
            slider("Max depth (mm)", "fused.tsdf.max_depth", lo=200, hi=1000, step=10, default=600, advanced=True, show_if="tsdf",
                   hint="Ignores anything farther from the scanner than this. Lower it if the background shows up."),
            slider("Pixel step", "fused.points.step", lo=1, hi=8, step=1, default=2, advanced=True, show_if="points",
                   hint="Uses every Nth pixel of each frame. Raise it for a faster, rougher build."),
            slider("Edge fade (px)", "fused.points.edge_px", lo=0, hi=30, step=1, default=6, advanced=True, show_if="points",
                   hint="Skips this many pixels next to depth edges, where readings are noisy. Raise it if edges look fuzzy."),
            slider("Min. frames", "fused.points.min_frames", lo=1, hi=50, step=1, default=3, advanced=True, show_if="points",
                   hint="Keeps a point only when this many frames saw it. Raise it to drop noise; lower it if thin parts vanish."),
            slider("Min. sides", "fused.points.min_sides", lo=1, hi=10, step=1, default=1, advanced=True, show_if="points",
                   hint="Keeps a point only when this many sides saw it. Raise it to drop stray bits only one side saw."),
            slider("Facing agreement", "fused.points.normal_agree", lo=0, hi=1, step=0.05, default=0.3, advanced=True, show_if="points",
                   hint="How much nearby points must face the same way to be kept. Raise it to drop rough noise.")],
             button=APPLY, run_to="fused",
             low_memory_warn="Below 0.25 mm, Standard blending needs a lot of memory."),
        dict(name="Isolation", label="Loose pieces", controls=[
            slider("Loose piece size", tool="isolation", param="rate", lo=0, hi=100, step=0.5, default={"points": 1, "mesh": 3},
                   revo=15, unit="%", hint="Removes floating bits smaller than this share of the biggest piece. Raise it if junk stays; lower it if real small parts go.")],
             button=DETECT, run_to="fused"),
        dict(name="Cut base", controls=[
            slider("Margin (mm)", tool="cut_plane", param="margin", lo=0, hi=20, step=0.5, default=1.5, unit="mm",
                   hint="Also removes this much above the layer you clicked: for scan noise and a thick base.")],
             button=None, run_to="fused"),
        dict(name="Overlap Detection", label="Overlaps", controls=[
            slider("Merge distance (mm)", tool="overlap", param="distance", lo=0.05, hi=2.0, step=0.01, default=0.21, revo=0.21,
                   unit="mm", fmt="{:.2f}", hint="Where passes left doubled layers closer than this, keeps one. Raise it if walls look doubled. Opposite faces never merge.")],
             button=DETECT, run_to="fused", only_kind="points",
             off_note="Pick Points under Blend method to use this. Standard already merges overlapping layers."),
        dict(name="Smooth", controls=[
            slider("Strength", tool="smooth", param="strength", lo=1, hi=25, step=1, default=10, revo=10, show_if="points",
                   hint="How far points are pulled onto their surface. Raise it if the surface looks rough."),
            slider("Passes", tool="smooth", param="times", lo=1, hi=50, step=1, default=2, revo=10, show_if="points",
                   hint="How many times to smooth. More is smoother but softens small detail."),
            slider("Strength", tool="smooth", param="strength", lo=5, hi=100, step=5, default=30, revo=30, unit="%", show_if="mesh",
                   hint="Softens noise without shrinking the part. Raise it if the surface looks bumpy. 30% is the usual."),
            slider("Passes", tool="smooth", param="times", lo=1, hi=20, step=1, default=3, revo=3, show_if="mesh",
                   hint="How many times to smooth. More is smoother but softens small detail.")],
             button=APPLY, run_to="fused")],
        more_tools_mesh=["fill_holes", "simplify", "slivers"]),     # the "..." at the end of the tab bar, mesh-kind fused only

    # the Mesh panel: four tabs
    "meshed": dict(title="Mesh", list_token="meshed", tabs=[
        dict(name="Mesh", controls=[
            seg("Surface", "meshed.method",
                [("auto", "Auto", "Picks the right one for how the scan was built."),
                 ("as_fused", "As scanned", "Keeps the surface from Fused as it is. Pick Watertight for one closed surface."),
                 ("poisson", "Watertight", "Wraps the scan in one smooth, closed skin, trimmed to where you scanned. Best for printing."),
                 ("bpa", "Joined points", "Joins the scanned points into triangles as they are. Holes stay open.")], "auto"),
            slider("Quality", "meshed.poisson.quality", lo=1, hi=7, step=1, default=6, revo=7, fmt="{:.1f}",
                   show_if="poisson", hint="Higher keeps finer detail but takes longer. Raise it if small details look melted. 7 is the finest.",
                   ),
            toggle("Close small holes", "meshed.poisson.hole_fill", False, revo=False, show_if="poisson",
                   hint="Closes holes next to the scanned surface, up to Fill reach away. Turn off if it closes openings the part should have."),
            slider("Trim (mm)", "meshed.poisson.trim", lo=0.1, hi=5, step=0.1, default=0.5, advanced=True, show_if="poisson",
                   hint="Cuts surface farther than this from the scanned points, so nothing is made up. Raise it if holes open up."),
            slider("Fill reach (mm)", "meshed.poisson.fill_trim", lo=0.5, hi=10, step=0.5, default=3.0, advanced=True, show_if="poisson",
                   hint="How far Close small holes may reach to patch a gap. Raise it to close bigger gaps."),
            slider("Drop pieces under (share)", "meshed.poisson.min_piece", lo=0, hi=0.1, step=0.001, default=0.002, advanced=True,
                   show_if="poisson", fmt="{:.3f}", hint="Removes loose bits smaller than this share of the model. Raise it if junk is left."),
            slider("Light smoothing passes", "meshed.poisson.taubin", lo=0, hi=20, step=1, default=3, advanced=True, show_if="poisson",
                   hint="Softens noise after the surface is made. 0 = off."),
            seg("Builder", "meshed.poisson.backend", [("auto", "Auto", ""), ("pml", "PyMeshLab", ""), ("o3d", "Open3D", "")],
                "auto", hint="Which program makes the surface. Keep Auto; switch only if building fails.", advanced=True, show_if="poisson"),
            slider("Point weight", "meshed.poisson.pointweight", lo=0, hi=20, step=0.5, default=4, advanced=True, show_if="poisson",
                   hint="How tightly the surface hugs the points. Raise it for sharper detail; lower it for a smoother surface."),
            slider("Samples per node", "meshed.poisson.samples_per_node", lo=1, hi=20, step=0.5, default=1.5, advanced=True, show_if="poisson",
                   hint="Points needed before finer detail is added. Raise it for a smoother surface on noisy scans."),
            toggle("Confidence weights", "meshed.poisson.confidence", True, advanced=True, show_if="poisson",
                   hint="Gives clearly seen points more say in the surface. Leave on unless the surface looks wrong.")],
             button=APPLY, run_to="meshed",
             grid="(about {grid:.2f} mm detail)",          # after "Quality" and its reset arrow, from PR.poisson_plan(extent, q)
             grid_same="(about {grid:.2f} mm, the finest here)",            # this quality gives the same grid as {same}: the cap for this part
             grid_unknown="(detail known once Fused is built)",
             hint="",                                       # poisson_off says it when it matters
             poisson_off="Pick Watertight above to use Quality and Close small holes."),
        dict(name="Isolation", label="Loose pieces", controls=[
            slider("Loose piece size", tool="isolation", param="rate", lo=0, hi=100, step=0.5, default=3, revo=15, unit="%",
                   hint="Removes floating bits smaller than this share of the biggest piece. Raise it if junk stays; lower it if real small parts go.")],
             button=DETECT, run_to="meshed"),
        dict(name="Cut base", controls=[
            slider("Margin (mm)", tool="cut_plane", param="margin", lo=0, hi=20, step=0.5, default=1.5, unit="mm",
                   hint="Also removes this much above the layer you clicked: for scan noise and a thick base.")],
             button=None, run_to="meshed"),
        dict(name="Simplify", controls=[
            slider("Triangles to keep", tool="simplify", param="keep_pct", lo=1, hi=99, step=1, default=40, revo=40, unit="%",
                   hint="Keeps this share of the triangles: {before} to about {after}. Lower it for a smaller file.") | dict(hint_plain="Keeps this share of the triangles. Lower it for a smaller file.")],
             button=APPLY, run_to="meshed"),
        dict(name="Smooth", controls=[
            slider("Strength", tool="smooth", param="strength", lo=5, hi=100, step=5, default=30, revo=30, unit="%",
                   hint="Softens noise without shrinking the part. Raise it if the surface looks bumpy."),
            slider("Passes", tool="smooth", param="times", lo=1, hi=20, step=1, default=3, revo=3,
                   hint="How many times to smooth. More is smoother but softens small detail.")],
             button=APPLY, run_to="meshed")],
        # our own tool: a block of its own under the tabs, always in sight on the Mesh panel
        blocks=[dict(name="Orient", title="Orient for export", controls=[], button=None,
                     intro="Choose which way is up and which side is the front. It is used when you export and when you look at the model. Nothing rebuilds."),
                dict(name="Finish for print", title="Finish for print", controls=[
            slider("Target triangles", tool="finish", param="faces", lo=100000, hi=1000000, step=10000, default=250000, fmt="k",
                   hint="Fewer triangles make a smaller file; too few flattens curves. {now}") | dict(hint_plain="Fewer triangles make a smaller file; too few flattens curves."),
            slider("Smoothness", tool="finish", param="smooth", lo=0, hi=20, step=1, default=5,
                   hint="Softens scan noise without shrinking the part. Raise it if the print looks rough. 0 = off."),
            slider("Fill holes up to", tool="finish", param="hole_edges", lo=0, hi=2000, step=50, default=400,
                   hint="Closes holes up to about this size across. Raise it if holes are left. 0 = leaves them open."),
            slider("Detail level", tool="finish", param="depth", lo=6, hi=11, step=1, default=9, advanced=True,
                   hint="Higher follows the scan closer but needs more memory. Raise it if detail is missing."),
            slider("Keep surface within (mm)", tool="finish", param="trim_mm", lo=0.5, hi=10, step=0.5, default=3.0, advanced=True,
                   hint="New surface farther than this from the scan is cut away. Lower it if blobs appear; raise it if holes reopen.")],
            button=APPLY, run_to="meshed",
            intro="Makes one smooth, closed, lighter model for your slicer as the last step. Press Apply to make it.",
            stl="( about {mb} MB STL )",                   # after "Target triangles": 50 bytes a triangle + 84
            now="The model has {faces} now.",
            holes_mm="{mm:g} mm", holes_edges="{edges} edges", holes_off="Off",
            extra=[("Export STL", "Save the model as a binary STL file for your slicer")])],
        more_tools_mesh=["fill_holes", "slivers"]),

    "report": dict(title="What is left", tabs=[
        dict(name="Map", controls=[], button=None,
             head="Open edge: {edge_mm} mm in {places} places",
             sub="Pinholes: {pinholes} (holes under {min_mm} mm, left to clean-up)",
             legend=[("lost", "Missing"), ("warn", "Rough"), ("ink3", "Fine")],
             row="{k}    {mm} mm of open edge", row_tip="Show this place",
             extra=[("Scan a new side", "Go to the Scan step with a new side ready")]),
        dict(name="Settings", controls=[
            slider("Ignore holes shorter than (mm)", "report.min_mm", lo=0, hi=100, step=1, default=8.0,
                   hint="Holes with less open edge than this are not listed. Raise it to see only the big gaps.")],
             button=APPLY, run_to="report")]),
}

# the "..." tools (mesh results only), same panel look as a tab
MORE_TOOLS = {
    "local_fix": dict(name="Smooth area", label="Smooth area", controls=[
        dict(kind="combo", label="Action", tool="local_fix", param="action", default="smooth", choices=[("smooth", "Smooth the area"), ("fill", "Fill a gap (a guess)")]),
        dict(kind="combo", label="Shape", tool="local_fix", param="shape", default="cylinder", choices=[("cylinder", "Cylinder"), ("box", "Box")]),
        slider("Diameter (mm)", tool="local_fix", param="diameter", lo=1, hi=150, step=0.5, default=20.0),
        slider("Box width (mm)", tool="local_fix", param="width", lo=1, hi=150, step=0.5, default=20.0),
        slider("Box height (mm)", tool="local_fix", param="height", lo=1, hi=150, step=0.5, default=20.0),
        slider("Depth (mm)", tool="local_fix", param="depth", lo=0.5, hi=60, step=0.5, default=6.0),
        slider("Fade at the edge (mm)", tool="local_fix", param="feather", lo=0, hi=50, step=0.5, default=4.0, hint="How gently the change blends into the surface around it. Raise it if a ridge shows at the edge."),
        slider("Strength", tool="local_fix", param="strength", lo=0.1, hi=1.0, step=0.05, default=0.8, fmt="{:.2f}", hint="How strongly each pass smooths. 1 is full strength."),
        slider("Passes", tool="local_fix", param="passes", lo=1, hi=50, step=1, default=8, hint="How many times to smooth. Raise it if bumps are still there."),
        slider("Reach (mm)", tool="local_fix", param="reach", lo=0.2, hi=15, step=0.1, default=2.5, fmt="{:.1f}", hint="How wide a bump it flattens. Bigger smooths wider bumps.")]),
    "clone": dict(name="Clone", label="Clone", controls=[
        dict(kind="combo", label="Shape", tool="clone", param="shape", default="cylinder", choices=[("cylinder", "Cylinder"), ("box", "Box")]),
        slider("Diameter (mm)", tool="clone", param="diameter", lo=1, hi=150, step=0.5, default=20.0),
        slider("Box width (mm)", tool="clone", param="width", lo=1, hi=150, step=0.5, default=20.0),
        slider("Box height (mm)", tool="clone", param="height", lo=1, hi=150, step=0.5, default=20.0),
        slider("Depth (mm)", tool="clone", param="depth", lo=0.5, hi=60, step=0.5, default=8.0),
        slider("Rotation (deg)", tool="clone", param="rotation", lo=-180, hi=180, step=1, default=0.0),
        dict(kind="combo", label="Mirror", tool="clone", param="mirror", default="no", choices=[("no", "No"), ("yes", "Mirror it")])]),
    "drill": dict(name="Cut shape", controls=[
        dict(kind="combo", label="Shape", tool="drill", param="shape", default="cylinder", choices=[("cylinder", "Cylinder"), ("box", "Box"), ("sphere", "Sphere")]),
        slider('Diameter (mm)', tool="drill", param='diameter', lo=0.2, hi=100, step=0.1, default=5.0),
        slider('Box width (mm)', tool="drill", param='width', lo=0.2, hi=100, step=0.1, default=5.0),
        slider('Box height (mm)', tool="drill", param='height', lo=0.2, hi=100, step=0.1, default=5.0),
        slider('Depth (mm)', tool="drill", param='depth', lo=0.1, hi=500, step=0.1, default=10.0),
        slider('Tilt (degrees)', tool="drill", param='tilt', lo=-89, hi=89, step=0.1, default=0.0, advanced=True),
        slider('Rotation (degrees)', tool="drill", param='rotation', lo=0, hi=360, step=0.1, default=0.0, advanced=True),
        slider('Move across (mm)', tool="drill", param='offset_u', lo=-100, hi=100, step=0.1, default=0.0, advanced=True),
        slider('Move along (mm)', tool="drill", param='offset_v', lo=-100, hi=100, step=0.1, default=0.0, advanced=True),
        slider('Move inward (mm)', tool="drill", param='inset', lo=-100, hi=500, step=0.1, default=0.0, advanced=True)],
        hint="Pick a spot on the surface. Red is what Apply removes. Diameter sets round shapes; width and height set Box. Depth sets how deep it goes.", button=APPLY,
        extra=[("Pick cut spot", "Pick a surface spot for the cut")]),
    "slice_gap": dict(name="Cut gap", controls=[
        slider("Gap width (mm)", tool="slice_gap", param="width", lo=0.1, hi=100, step=0.1, default=1.),
        slider("Cut depth (mm)", tool="slice_gap", param="depth", lo=0.1, hi=500, step=0.1, default=5.)],
        hint="Pick both ends of the join to cut. Red marks the slot. Apply cuts it and closes the cut faces, so the gap stays open.", button=APPLY,
        extra=[("Pick cut ends", "Pick two surface spots along the unwanted join")]),
    "fin": dict(name="Add fin", controls=[
        slider("Thickness (mm)", tool="fin", param="thickness", lo=0.2, hi=30, step=0.1, default=2.),
        slider("Reach (mm)", tool="fin", param="reach", lo=0.5, hi=100, step=0.5, default=8.),
        dict(kind="combo", label="Shape", tool="fin", param="shape", default="triangle", choices=[("triangle", "Triangular"), ("rounded", "Rounded")])],
        hint="Pick the two ends, then a third spot to set the fin plane. More spots refine the plane. Red shows the fin before Apply.", button=APPLY,
        extra=[("Pick attachment spots", "Pick two new spots for the fin")]),
    "mirror": dict(name="Mirror", label="Mirror", controls=[
        dict(kind="combo", label="Plane", tool="mirror", param="fit", default="yes", choices=[("yes", "Refine it to the part"), ("no", "Exactly as picked")]),
        slider("Fill where the other side has nothing within (mm)", tool="mirror", param="gap_mm", lo=0.2, hi=20, step=0.1, default=1.5, fmt="{:.1f}",
               hint="Smaller adds more of the copy (closer to what is there); bigger keeps more of the measured side alone.")]),
    "fill_holes": dict(name="Hole Filling", label="Fill holes", controls=[
        slider("Largest hole (mm)", tool="fill_holes", param="hole_mm", lo=0, hi=500, step=1, default=0, fmt="{:g}",
               hint="Closes holes up to this size across. Raise it if holes are left. 0 = Auto, about 4% of the part's size.")], button=APPLY),
    "simplify": dict(name="Simplify", controls=[
        slider("Triangles to keep", tool="simplify", param="keep_pct", lo=1, hi=99, step=1, default=40, revo=40, unit="%", hint="Keeps this share of the triangles. Lower it for a smaller file.")], button=APPLY),
    "slivers": dict(name="Remove slivers", controls=[
        slider("Longer than (x usual edge)", tool="slivers", param="factor", lo=1.5, hi=20, step=0.5, default=4, hint="Removes thin triangles this many times longer than a usual edge. Lower it to remove more.")], button=APPLY),
}

# ------------------------------------------------------------------------------------------------ "Applied" list (under the panel)
def operation_stats(tool, stats, kind):
    """Describe what the operation did, using its result rather than its name."""
    if "moved" in stats: return "Moved {:,} points".format(stats["moved"])
    if tool == "local_fix" and "added_faces" in stats: return "Filled {:,} faces".format(stats["added_faces"]) + (" over {:,.1f} mm2".format(stats["filled_mm2"]) if "filled_mm2" in stats else "")
    if "added_points" in stats: return "Added {:,} points".format(stats["added_points"])
    if "added_faces" in stats: return "Added {:,} faces".format(stats["added_faces"])
    if "added" in stats: return "Added {:,} {}".format(stats["added"], "points" if kind == "points" else "faces")
    if "removed" in stats: return "Removed {:,} {}".format(stats["removed"], "points" if kind == "points" else "faces")
    return ""


APPLIED = dict(
    head="Applied to {stage}",                  # stage = Fused / Meshed / Side 3 / Line-up
    empty="Nothing applied yet. Tools you apply stack here, in order.",
    row="{k}  {label} {summary}",                # "2  Smooth 10 x10"; summary from Tool.summary(params)
    row_stats="removed {removed} {what}",        # what = points / polygons
    row_stats_smooth="smoothed, dropped {removed} stray {what}",   # Smooth drops points with too few neighbours
    edit="Edit", on="Turn on", off="Turn off", remove="Remove",
    parked="Kept for {method}: {n} tool{s}",     # ops of a fused method not in use (engine: parked)
    deletes="Deletes, kept for every blend method", # the Fused stage's own delete list
    check="Check: took {now} {what}, was {was}",
    gone="Already gone: an earlier delete took this area",
    editing="Editing step {k}: change it and press Update. Later steps run again.",
)

# ------------------------------------------------------------------------------------------------ detect
DETECTED = dict(
    already="Already removed at {now:g}%\n{pieces} piece{s} under {rate:g}% are gone. Nothing to delete.",
    also="{pieces} piece{s} marked · {count:,} {what} ({pct:.1f}%)\n{gone} more already removed at {now:g}%.",
    count="{count} of {n} {what} would go ({pct:.1f}%)",     # what = points / polygons
    pieces="{pieces} piece{s} would go: {count} of {n} {what} ({pct:.1f}%)",   # Isolation: pieces under the rate
    overlap_count="{count} of {n} points would merge away ({pct:.1f}%)",
    none="Nothing would go at this setting.",
    back="Lower than now ({now:g}%): {count} {what} come back.",   # Side Isolation below the side's rate
    moved="Detect again to see what goes.",                  # slider moved after a Detect: the red clears
    chip="Red: would go",                                    # colour key on the view while detected
)

# ------------------------------------------------------------------------------------------------ bottom bar
BOTTOM = dict(
    pager="{k} / {n}",                          # "< 1 / 4 >"; shows "Side" above it in small text
    pager_label="Side",
    pager_all="All sides",                      # stages after Side: the pager is replaced by this dim text
    rect="Rectangle", lasso="Lasso",            # round buttons with the label under them
    through="Through",                          # checkbox: also select what is hidden behind (default off: front only)
    selected="{count} {what} selected",
    sel_delete="Delete", sel_invert="Invert", sel_clear="Clear",
    undo_tip="Undo: {note}", redo_tip="Redo: {note}", nothing_undo="Nothing to undo", nothing_redo="Nothing to redo",
    history="History", history_now="now",
)

REBUILD_BAR = dict(
    text="{changed} changed. {rest} will rebuild (about {eta}).",   # rest = "Meshed and Coverage"
    plain_text="{rest} will rebuild (about {eta}).",               # nothing changed here: something the build uses did
    fresh_text="Not built yet \u00b7 about {eta}",                  # a scan never built: nothing to "rebuild"
    pending_text="{n} delete{s} waiting. The picture shows them gone; Rebuild makes it real.",   # Edit deletes wait for one rebuild (2026-10-25)
    pending_button="Rebuild",
    fresh_button="Build it",
    button="Rebuild the rest",
    undo_text="After the undo, {rest} will rebuild (about {eta}).",
    model_note="The saved model file updates when Coverage is built.",
)

# ------------------------------------------------------------------------------------------------ progress card--------------
PROGRESS = dict(
    fusing="Blending the scan",                   # the title while Fusion runs
    processing="Working",                       # the title for any tool
    onetap="YOLO Yoink",                        # the title during YOLO Yoink
    line1="Now: {what}",                        # "Now: Fusion"
    line2="Settings: {params}",                 # "Settings: 0.4 mm", "Settings: 7.0"
    stop="Stop", stopping="Stopping...", elapsed="{m}:{s:02d}", starting="Starting",
    cached="{n} step{s} already built, reused",
    fun=("Reticulating splines", "Herding the point cloud", "Teaching dots to line up", "Polishing every point", "Counting triangles twice",
         "Warming up the globe", "Yoinking points into place", "Untangling the scan", "Asking the turntable nicely", "Nudging stray points home"),
    overall="Step {k} of {n}",                  # steps this run does (not the reused ones)
    overall_many="{done} of {n} steps done, {active} running",      # sides building side by side
    left="about {eta} left",
    doing="Running: {stages}",                  # "Side, Line-up, Fused, Meshed and Coverage"
    failed="{stage} failed: {why}",
    stopped="Stopped. Nothing half-built was kept.",
)
# what = the node in plain words; params = its key values
PROGRESS_WHAT = {
    "side.fuse": ("Side {n}: Fusion", "{voxel} mm"),
    "side.platter": ("Side {n}: Cut base", "{method}"),
    "side.keep": ("Side {n}: Isolation", "{min_share_pct}%"),
    "lineup.pairs": ("Line-up: pair {k} of {n}", "{voxel} mm"),
    "lineup.place": ("Line-up: placing sides", "base {base}"),
    "lineup.fix": ("Line-up: flips", "{count}"),
    "lineup.refine": ("Line-up: refining", "{rounds} rounds"),
    "lineup.adjust": ("Line-up: nudges", "{count}"),
    "fused": ("Fusion", "{point_distance} mm"),
    "op.isolation": ("Isolation", "{rate}%"),
    "op.overlap": ("Merging overlaps", "{distance} mm"),
    "op.smooth": ("Smooth", "{strength}, {times}"),
    "op.simplify": ("Simplify", "{keep_pct}%"),
    "op.fill_holes": ("Filling holes", "{hole_mm_or_auto}"),
    "op.slivers": ("Remove slivers", "{factor}"),
    "op.finish": ("Finish for print", "{faces} triangles"),
    "op.delete": ("Delete", "{shape}"),
    "meshed": ("Mesh", "{quality:.1f}"),       # as_fused: ("Mesh", "as fused")
    "report": ("Coverage", "{red_mm} mm, {yellow_mm} mm"),
}

# ------------------------------------------------------------------------------------------------ YOLO Yoink
ONETAP = dict(
    button="YOLO Yoink",
    tip="Builds every step with a preset, start to finish. Your own deletes are kept.",
    title="YOLO Yoink",
    preset_label="Preset",
    presets=[("today", "Classic Yoink", "Standard blend at 0.4 mm, 300 frames a side, drops loose pieces under 3%."),
             ("revo", "Detail Yoink", "Points blend at 0.4 mm, loose pieces 1%, merge overlaps 0.21 mm, smooth 10 x2, Watertight quality 6."),
             ("revo-device", "Deep Clean", "Loose pieces 15%, smooth 10 x10, quality 7. Removes more loose bits and softens the surface."),
             ("plus", "Sharp Yoink", "Plus blend for a sharp surface and more detail."),
             ("quick", "Quick Peek", "Standard blend at 0.8 mm, 100 frames a side. A fast look while scanning."),
             ("print", "Print Yoink", "Plus blend at 0.4 mm, then Finish for print: one smooth, closed model of 250k triangles, about 12.5 MB as STL.")],
    # any other PR.PRESETS name gets a chip with its note as the hint (the plus preset, once it lands)
    built="Already built: nothing to run.",
    cancel="Cancel", ok="Yoink it",
)

# ------------------------------------------------------------------------------------------------ recipe chip (under YOLO Yoink)
RECIPE = dict(
    chip="Recipe: {name}",
    edited="{name}, edited",
    tip="What Build uses. Pick another recipe here; nothing builds until you press Build.",
    menu_hint={                   # what each is best at, measured in the 2026-10-13 bake-off (one scan, and a 12-side scan)
        "today": "Keeps fine detail and true depth, like raised text",
        "quick": "Fast check, low detail, can leave holes",
        "revo": "Very smooth new surface, fine text softens",
        "revo-device": "Like Detail with more cleanup, biggest file",
        "plus": "Closest to the scanner's own model, smooth on many sides",
        "print": "Small closed file ready to slice, fine text softens",
    },
    card_sub={                    # the card's second line (short: two cards share a row)
        "today": "Fine detail, true depth", "quick": "Fast check, low detail", "revo": "Very smooth, text softens",
        "revo-device": "Smoothest, most cleanup", "plus": "Closest to the scanner", "print": "Small, closed, ready to slice",
    },
    card_facts={                  # the third line: a measured fact (12-side test scan)
        "today": "About 50 MB, noisy on many sides", "quick": "About 11 MB, a fraction of the time",
        "revo": "About 120 MB, slower", "revo-device": "About 155 MB, slower",
        "plus": "About 45 MB, best on many sides", "print": "About 5 MB, fully closed",
    },
    all_options="All options and cards...",
    ultra="Ultra Yoink: build several and compare...",
    picked="Recipe: {name}. Press Build to use it.",
    build="Build \u00b7 {name}", rebuild="Rebuild \u00b7 {name}",
)

ULTRA = dict(
    title="Ultra Yoink",
    intro="Builds every recipe you tick, one after another, then lets you flip between the results and keep the best. "
          "The sides and line-up build once; each recipe then only builds its own Fused and Meshed steps.",
    loading="Working out the time and disk space...",
    plan_failed="Could not work out the plan: {why}",
    col_recipe="Recipe", col_time="Time", col_disk="Disk", col_took="Took", col_faces="Triangles", col_size="Model",
    built="already built",
    warn="Builds {n} recipes: {time} and about {disk} more on disk (free: {free}). Every result is kept, so switching "
         "between them later is instant. Delete old versions in Versions to get the space back.",
    none="Tick at least one recipe.",
    go="Ultra Yoink it", cancel="Close", running_button="Building...",
    running="Building {done} of {n} done. Now: {current}. You can close this; it keeps going.",
    results_intro="Show one to look at it in the view (it is stored, so it appears in seconds). The one you show last "
                  "stays as the project's recipe.",
    show="Show", failed="Failed", again="Run again...",
    done="All done. Show each one and keep the one you like.",
    stopped="Stopped early. The results built so far are below.",
)

# ------------------------------------------------------------------------------------------------ Export STL (Finish for print block)
EXPORT = dict(
    title="Export STL", filter="STL files (*.stl)", suffix="_print.stl",     # default: <project>/<project name>_print.stl
    saving="Saving {name}...",
    done="Saved {name}: {mb} MB, {faces} triangles.",
    failed="Export STL failed: {why}",
    nothing="Build the model first: there is nothing to export yet.",
)

# ------------------------------------------------------------------------------------------------ Export sheet (File > Export..., the ribbon's Export)
EXPORT_SHEET = dict(
    title="Export model", format="File type", cancel="Cancel", ok="Export",
    formats=(("stl", "STL"), ("obj", "OBJ"), ("ply", "PLY"), ("3mf", "3MF")),
    hints={"stl": "For your slicer. Shape only, no color.",
           "obj": "For 3D apps like Blender. Keeps color when the model has it.",
           "ply": "Keeps color when the model has it. Opens in MeshLab and CloudCompare.",
           "3mf": "For newer slicers. One file, sized in mm."},
    zip="Add the scan data (one ZIP file)",
    zip_hint="The model plus every side's scan folder, to back up the scan or share it. This file can be large.",
    zip_none="This project has no scan folders to add.",
    measuring="Size: measuring...", size="Size: {dims}",
    file_measuring="File: working it out...", file="File: about {size}", zip_file="ZIP: about {size} (the model and the scan data)",
    save_title="Export {fmt}", filter="{fmt} files (*.{ext})", zip_filter="ZIP files (*.zip)",
    suffix="_print", zip_suffix="_scan",               # <project name>_print.stl, <project name>_scan.zip
    closed_checking="Closed surface: checking...", closed_ok="Closed surface: yes. Slicers take it as is.",
    closed_no="Not closed: {n} open edges. Slicers will ask for a repair. Watertight in Meshed closes it.",
    closed_unknown="",
    make_watertight="Make watertight and rebuild",
    done="Saved {name}: {mb} MB, {faces} triangles, {dims}.",
    zip_done="Saved {name}: {mb} MB, the model ({faces} triangles, {dims}) and {sides} scan folder{s}.",
    failed="Export failed: {why}",
)

# ------------------------------------------------------------------------------------------------ view overlay
VIEW = dict(
    empty="Press Rebuild at the top, or Apply, to build this step. Nothing to show yet.",
    empty_busy="Building. The result shows here when it is done.",
    opening="Opening the project...",
    ghosts_old="Gray sides: last build",
    home="Home view",
    compare="Compare",
    before="Before", after="After",
    compare_menu=[("tool", "Before the last tool"), ("stage", "Before all tools"), ("version", "An earlier version...")],
    before_short={"tool": "last tool", "stage": "all tools", "change": "this change", "version": "version"},   # on the Before button
    all_versions="All versions...", version_from="Version from {when}",
    peek="Hold B to see before",
    quality="Quality",
    quality_wait="Measuring how well each spot was scanned. About 30 s the first time.",
    quality_done="Quality: red {lo}% (few frames: rough), yellow {mid}%, blue {hi}% (many frames: smooth). Scan more where it is red.",
    quality_fail="Could not measure the quality: {why}",
    colour="Color", colours=[("shaded", "Shaded"), ("sides", "Sides"), ("support", "Support"), ("scanhere", "Coverage"),
                             ("flat", "Flat facets")],
    platter_cut="Violet: base cut away",
    cut_big="Lower the Margin or pick the base by hand if the part is not that small: this cut removes {pct}% of this side.",
    chip="{stage} \u00b7 {summary}",            # the view's label: "Fused · 6 sides", "Built model · 6 sides"
    built="Built model", finished="Finished for print", triangles="{n} triangles", nothing_yet="not built yet", showing_old="showing the last build",
    published="from Build combined model, read only",
    lineup_chip="{n} sides, a color per side",
    not_built="Not built yet. Press Apply here, or YOLO Yoink.",
    left_out="Tick Use in Line-up to bring side {n} back. It is left out of the model, so it is not built.",
    not_built_raw="The frames are saved, they just have not been built into a picture yet. YOLO Yoink builds everything.",   # every listed side has frames
    not_built_edit="Nothing to edit yet. Build the model on the Build step first.",   # Edit: no Apply, no YOLO Yoink
    stale_chip="Out of date: built before {changed} changed",
    loading="Loading {what}...",
    platter_chip="Turntable: {source}, tilt {tilt:.1f} deg",   # source = found while scanning / searched / manual / off
)

VERSIONS = dict(
    title="Versions of {stage}",
    row="{when}    {summary}    {count}",       # summary "Standard 0.4 mm", count "1832709 polygons"
    compare="Compare", keep="Keep", unkeep="Don't keep",
    keep_tip="Kept versions are never deleted to free disk space.",
    current="in use",
    help="Each time this stage is built again, the earlier result is kept here. In use is what the model uses now. Compare shows an "
         "earlier one beside it in the view. Keep stops a version from being deleted when stored results fill their disk space.",
)

# ------------------------------------------------------------------------------------------------ refusals and notices
NOTICE = dict(
    nothing_box="Nothing under the box or loop. Faces seen edge-on count as hidden: set Reach to Through, or turn the model to face them.",
    nothing_click="No piece under the click. Click on the model itself.",
    delete_pending="Deleted. {n} delete{s} wait for one rebuild: keep going, then press Rebuild or leave Edit.",
    rebuild_now="Rebuilding your deletes first. Try that again when it finishes.",
    rebuild_first="The model is not built yet. Press Build combined model (or Rebuild in Edit) first.",
    recording="Pause the scan first: building needs the graphics card the scan is using.",
    scan_build="A build started on the Scan step is running. Tools come back when it is done.",
    locked="The project is being built somewhere else.",
    no_sides="No saved side yet: scan one on the Scan step first.",
    as_fused_points="Pick Standard under Blend method to use As scanned: Points make no surface of their own.",
    one_side="At least one side must stay in.",
    denied="This folder is read only for builds.",
    stopped="Stopped. Nothing half-built was kept.",
    out_of_date="Press Detect again: the view was out of date.",
    home_saved="This view is the Home view now, in every view of this project.", home_reset="Home view is back to the default.", home_not_saved="Could not save the Home view.",
    auto_out="{who} left out for not lining up with the others ({agree}% match). Often the base was not cut: fix it in Side, Cut base, Manual, then turn Use back on in Line-up.",
    latest_only="Editing works on the latest result",
    run_first="Build up to here first",
    too_big="Too big to edit at full size here: showing a lighter copy. Tools are off.",
    changed_under="Select again: the result changed under the selection.",
    took_other="Look it over, or undo: the delete took {got} instead of {want}.",
    scripts_changed="The build scripts changed, so every stage will rebuild (about {eta}).",
    unknown="That step or setting is not known to the build.",
    no_gl="Tools need the 3D view, which this computer could not open (no OpenGL). This is only a picture.",
    engine_error="The build engine said: {why}",
    tools_off_run="Selection is off while building.",
    loading="Loading the build...",
    no_picture="No picture of this stage yet",
    compare_off="Tools are off while comparing.",
    published="This is the model from Build combined model. Rebuild the rest to edit it here.",
    not_started="Try again in a moment: the build did not start because the scan session is busy.",
    detect_last="Only the last step can Detect again.",
    none_yet="None yet.",
    no_versions="No stored versions yet.",
    more_places="and {n} more",
    out_of_date_chip="Out of date",
)

# one colour per side (Line-up swatches, the "sides" colour mode); meshes, back faces and points use the theme's mesh,
# backface and cloud tokens
SIDE_COLOURS = ["#e8744f", "#5b9df0", "#4cc86a", "#f2b43a", "#b27cff", "#4fd1c5", "#f06292", "#a3e635", "#fb923c", "#93c5fd",
                "#c084fc", "#facc15", "#34d399", "#f87171", "#60a5fa", "#e879f9"]
# one colour per point pair on the Line-up Points tab; residual colours: kept < 5 mm, warn > 15 mm
PAIR_COLOURS = ["#ff5252", "#40c4ff", "#69f0ae", "#ffd740", "#e040fb", "#ff9e40", "#18ffff", "#b2ff59"]
PAIR_GOOD_MM, PAIR_BAD_MM = 5.0, 15.0

# ------------------------------------------------------------------------------------------------ keys (Build page only)
KEYS = [
    ("Up / Down", "Previous / next stage"),
    ("Left / Right", "Previous / next side (Raw frames, Side)"),
    ("Ctrl+Enter", "Apply, Detect or Delete (the panel's button)"),
    ("Ctrl+Z", "Undo"), ("Ctrl+Y / Ctrl+Shift+Z", "Redo"),
    ("R", "Rectangle"), ("L", "Lasso"), ("T", "Through: also select what is hidden behind"),
    ("Del", "Delete the selection"), ("Esc", "Clear the selection or the red"),
    ("H", "Home view"), ("C", "Compare"), ("B (hold)", "See before"),
    ("Ctrl+B", "Rebuild the rest"), ("Ctrl+Shift+B", "YOLO Yoink"), ("Ctrl+1 to 3", "Go to a step"),
    ("?", "This sheet (Esc closes it)"),
]



def tab_of(panel, name):
    """A panel's tab dict by its name (not by position: tabs get added)."""
    return next(t for t in PANELS[panel]["tabs"] if t.get("name") == name)


# ------------------------------------------------------------------------------------------------ engine warnings, plainly
# (pattern, plain words): the engine's own text stays in the build log file; the app shows these. {1}, {2} are the pattern's groups.
PLAIN_WARN = [
    (r"Side (\d+): the base plane found runs through the part", "Side {1}: Auto could not find the base, so the side was kept uncut. Fix it in Cut base, Manual."),
    (r"plate top rejected \(", "No plate found under the part, so the turntable was used instead."),
    (r"plate top search failed \(", "The plate search failed, so the turntable was used instead."),
    (r"Side (\d+): its base plane was ignored for placing", "Side {1}: its base looked wrong, so Line-up placed it by shape alone."),
    (r"table cut failed.*(?:using (?:it|the side) uncut)", "The base cut failed, so the side was kept uncut. It is tried again on the next build."),
    (r"GPU was requested, but fusion ran on the CPU", "Built on the CPU instead of the graphics card. The result can differ a little."),
    (r"failed the same way (\d+) runs in a row", "This step failed the same way {1} times in a row, so it is left as it is."),
    (r"will retry next run", "This step is tried again on the next build."),
    (r"no PyMeshLab python configured", "Used the built-in surface builder."),
    (r"(?:tsdf|plus) below 0\.25 mm needs a lot of", "Very small point spacing: this needs a lot of memory."),
    (r"free-space votes reach less than", "Thin edges may close up at this setting."),
    (r"scan-here map failed", "The coverage map could not be made this time."),
    (r"Cleanup backend changed during the job", "Settings changed during cleanup, so this result will be rebuilt."),
]


def plain_warning(text):
    """An engine warning in words a first-timer can act on; the text itself when there is no plain version."""
    import re
    t = str(text)
    for pat, words in PLAIN_WARN:
        m = re.search(pat, t)
        if m: return words.format(*([""] + list(m.groups())))
    return t

TIMELAPSE_SHEET = dict(
    title="Export scan timelapse", open_timelapse="Scan timelapse...",
    intro="A short clip of the model growing as the scan went, made from the saved frames. Good for sharing.",
    sides="Sides", all_sides="All sides", side="Side {n}",
    layout="Show", layouts=(("model", "Model"), ("camera+model", "Camera and model")),
    size="Size", sizes=(("480p", "480p"), ("720p", "720p"), ("1080p", "1080p")),
    length="Length", lengths=((5, "5 s"), (8, "8 s"), (15, "15 s"), (30, "30 s")),
    format="File type", formats=(("mp4", "MP4"), ("gif", "GIF")),
    hints={"mp4": "Small and sharp. Plays almost anywhere.",
           "gif": "Plays in chats and on web pages. A bigger file with fewer colors."},
    no_mp4="MP4 needs ffmpeg, which is not installed here. GIF works without it.",
    orbit="Move camera while the model grows", mark="PointYoink mark in the corner",
    save_to="Saves as {name} in the project folder.",
    working="Making the clip... {pct}%", starting="Starting...",
    cancel="Cancel", ok="Export", stop="Stop",
    done="Saved {name}, {mb} MB.", failed="Timelapse failed: {why}", stopped="Timelapse stopped. Nothing was saved.",
    open_folder="Open folder",
    no_sides="This project has no saved sides yet. Scan a side first.",
)
