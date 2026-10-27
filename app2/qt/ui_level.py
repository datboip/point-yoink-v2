# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simple and Advanced. Simple shows what a scan needs: Start, sides, the bed, the light, Object, Surface and Range.
Advanced adds every other control. Hidden controls keep their values, and Auto gain steers the gain while it is hidden."""
from app2.scan import settings

LEVELS = ('Simple', 'Advanced')
TIP = 'Simple shows only what a scan needs. Advanced shows every setting.'


def saved():
    level = settings.load().get('ui_level')
    return level if level in LEVELS else LEVELS[0]


def advanced_widgets(page):
    c = page.controls
    return (c.capture_label, c.capture_help, c.capture_mode, c.live_build, c.setup_section, c.calibration_setup,
            c.capture_tools, c.framing_row, c.frame_note, c.save_color, c.outside_label, c.outside_below, c.outside_help,
            c.below_bed, c.remove_base, c.gain_row, page.camera_options.ir, page.camera_options.show)


def apply(page, level):
    advanced = level == 'Advanced'
    page.advanced = advanced
    for widget in advanced_widgets(page): widget.setVisible(advanced)
    # Simple: the guide fills the room under the cameras that the setup rows leave; Advanced: top of Sides
    place_guide(page)
    page.live_strip.place_light()
    if not advanced:
        c = page.controls
        if c.auto_gain.isEnabled() and not c.auto_gain.isChecked(): c.auto_gain.click()   # nobody is left to set it by hand
        if page.camera_prefs.ir != 'Left': page.camera_prefs.set_ir('Left')               # the picker is hidden: show the plain picture
    for tile in (page.ir, page.depth, page.colour): tile.header.adapt(tile.width())
    page.controls.layout_dirty = True; page._controls_changed()


def place_guide(page):
    """Simple: under the cameras, keeping its room. Advanced, or a window too short to spare that room: top of Sides."""
    under = not getattr(page, 'advanced', True) and not page.live_strip.compact
    home = page.live_strip.layout() if under else page.sides.side_layout
    if home.indexOf(page.guide) != 0: home.insertWidget(0, page.guide)
    policy = page.guide.sizePolicy(); policy.setRetainSizeWhenHidden(under); page.guide.setSizePolicy(policy)
    if not under: page.guide.setMinimumHeight(0); page.guide.setMaximumHeight(16777215)
