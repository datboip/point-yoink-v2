# SPDX-License-Identifier: AGPL-3.0-or-later
"""PointYoink 2.0 look: colour tokens, fonts and the Qt style sheet.

Colours carry one meaning each, the same one everywhere (the IR tint, the rings, the status pill, the buttons, the
side cards, the sliders):
  kept     green   depth that is kept, tracking, Start
  lost     red     tracking lost, errors, "scan here" on the model
  warn     amber   paused, no frames, warnings, Pause
  platter  violet  the platter the build will cut, a base found from the bed
  accent   blue    what is picked or focused; the same blue as "well covered" on the model
Everything else is graphite, so the camera pictures and the model are the brightest things on screen. The IR
picture keeps the Tk scanner's own tint math (scan/preview.py); only the window's chrome uses these tokens."""
from PySide6.QtGui import QColor, QFont, QFontDatabase

T = dict(
    window="#1a1d22",      # app background
    panel="#20242a",       # rail, sides panel, controls
    raised="#2a2f37",      # buttons, cards, inputs
    raised_hi="#343a44",   # hover
    well="#0c0e11",        # behind camera pictures and the model: pictures read best on near-black
    line="#2f343c",        # 1 px separators
    ink="#e8eaed",         # text
    ink2="#a3aab3",        # secondary text
    ink3="#6c737c",        # hints, disabled
    kept="#4cc86a",
    kept_hi="#5ad777",
    lost="#ff5a4f",
    warn="#f2b43a",
    warn_hi="#f7c75e",
    platter="#b27cff",
    accent="#5b9df0",
    accent_hi="#78b0f5",
    on_color="#0d1014",    # text on a filled green/amber button
    mesh="#c4c8ce",        # Build step's 3D view: a mesh with no other colour (neutral, so red and blue read on it)
    backface="#5c546e",    # its back faces (an inside-out patch shows at a glance)
    cloud="#4f8fe8",       # points (blue)
)

INPUT_QSS = """
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    color: %(ink)s; background: #151a21; border: 1px solid #424b58;
    border-radius: 5px; padding: 3px 7px; selection-background-color: #274a78;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: %(accent)s; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: %(ink3)s; background: #20242a; border-color: %(line)s;
}
QComboBox { padding-right: 26px; }
QComboBox::drop-down { width: 22px; border-left: 1px solid #424b58; background: %(raised)s; border-top-right-radius: 4px; border-bottom-right-radius: 4px; }
QComboBox QAbstractItemView { background: %(raised)s; color: %(ink)s; selection-background-color: #274a78; border: 1px solid #424b58; }
QSpinBox, QDoubleSpinBox { padding-right: 22px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { width: 20px; subcontrol-origin: border; subcontrol-position: top right; background: %(raised)s; border-left: 1px solid #424b58; border-bottom: 1px solid #424b58; }
QSpinBox::down-button, QDoubleSpinBox::down-button { width: 20px; subcontrol-origin: border; subcontrol-position: bottom right; background: %(raised)s; border-left: 1px solid #424b58; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: %(raised_hi)s; }
""" % T

# Explicit glyphs stay visible when native spin and combo subcontrols are styled.
from pathlib import Path as _InputAssetPath
INPUT_QSS += """
QComboBox::down-arrow, QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("%(assets)s/input-down.svg"); width: 10px; height: 8px;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("%(assets)s/input-up.svg"); width: 10px; height: 8px;
}
QComboBox::down-arrow:disabled, QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {
    image: url("%(assets)s/input-down-disabled.svg");
}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off {
    image: url("%(assets)s/input-up-disabled.svg");
}
""" % {"assets": _InputAssetPath(__file__).resolve().with_name("assets").as_posix()}

UI_FAMILIES = ["Noto Sans", "Cantarell", "DejaVu Sans"]      # first one installed wins; Noto Sans has tabular digits,
                                                              # so fps / frame counts don't jitter as they change


def ui_font(px=13, weight=QFont.Weight.Normal):
    f = QFont(); f.setFamilies(UI_FAMILIES); f.setPixelSize(px); f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def qc(name, alpha=None):
    c = QColor(T[name])
    if alpha is not None: c.setAlphaF(alpha)
    return c


def qss():
    """The app-wide style sheet. Variants use a 'kind' property (primary, pause, build, chip, lamp, quiet, danger,
    float, seg) and containers a 'role' property, so no widget needs its own style string."""
    font_px=ui_font().pixelSize()
    scan_style=dict(T,scan_tab_y=font_px//2,scan_tab_x=font_px-font_px//4,scan_tool_pad=font_px//4,scan_corner=font_px//3)
    return """
QWidget { color: %(ink)s; background: transparent; }
QMainWindow, #Shell { background: %(window)s; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid %(ink2)s; border-radius: 3px; background: %(well)s; }
QCheckBox::indicator:checked { background: %(accent)s; border: 2px solid %(ink)s; }
QCheckBox::indicator:disabled { border-color: %(ink3)s; }
QToolTip { background: %(raised)s; color: %(ink)s; border: 1px solid %(line)s; padding: 4px 6px; }

QTabBar#ScanSettingsTabs::tab { background: %(raised)s; color: %(ink2)s; padding: %(scan_tab_y)dpx %(scan_tab_x)dpx; border: 1px solid %(line)s; }
QTabBar#ScanSettingsTabs::tab:selected { color: %(ink)s; border-bottom-color: %(accent)s; }
QFrame#ScanControls QToolButton { color: %(ink)s; background: %(raised)s; border: 1px solid %(line)s; border-radius: %(scan_corner)dpx; padding: %(scan_tool_pad)dpx; }
QFrame#ScanControls QToolButton:checked { background: %(kept)s; color: %(on_color)s; }
QFrame#ScanControls QToolButton:disabled { color: %(ink3)s; }
/* ---- left rail: the six steps ---- */
#Rail { background: %(panel)s; border-right: 1px solid %(line)s; }

/* ---- header ---- */
#Header { background: %(window)s; border-bottom: 1px solid %(line)s; }
#ProjectName { font-weight: 600; }
#SourcePill { background: %(raised)s; border-radius: 11px; padding: 2px 10px; color: %(ink2)s; }

/* ---- project-first Workspace (the calm entry point) ---- */
#WorkspaceHome { background: #1b1f25; }
#WorkspacePageHeader { background: %(window)s; border-bottom: 1px solid %(line)s; }
#WorkspaceTitle { color: %(ink)s; }
#WorkspaceStart { min-width: 430px; max-width: 820px; border-color: #4b627c; }
#WorkspacePrimary { background: %(accent)s; color: %(on_color)s; border: none; border-radius: 10px; font-weight: 700; text-align: center; }
#WorkspacePrimary:hover { background: %(accent_hi)s; }
#WorkspaceSecondary { background: %(raised)s; border: 1px solid #5b718a; border-radius: 10px; font-weight: 700; text-align: center; }
#WorkspaceSecondary:hover { background: %(raised_hi)s; }
#WorkspaceProjects { border-radius: 10px; }
#WorkspaceProjectCard { background: %(raised)s; border: 1px solid %(line)s; border-radius: 8px; }
#WorkspaceProjectCard:hover { background: %(raised_hi)s; border-color: %(accent)s; }
#WorkspaceProjectOpen { background: #223c5d; border: 1px solid #406c9d; border-radius: 7px; color: %(accent_hi)s; font-weight: 700; min-width: 68px; }
#WorkspaceProjectOpen:hover { background: #2b4d78; }
#WorkspaceProjectDelete { background: transparent; border: 1px solid #3a3f47; border-radius: 7px; color: #9aa5b4; min-width: 68px; }
#WorkspaceProjectDelete:hover { color: #ff8a80; border-color: #7a3b3b; }
#WorkspaceProjectDelete:disabled { color: #5a626c; }
#WorkspaceProjectDeleteYes { background: #5a2323; border: 1px solid #a33a3a; border-radius: 7px; color: #ffd4d0; font-weight: 700; min-width: 68px; }
#WorkspaceProjectDeleteYes:hover { background: #7a2e2e; }

#WorkspaceTools { background: #252a31; border-bottom: 1px solid %(line)s; }
#WorkspaceBuildTools { background: #20252c; border-left: 2px solid %(accent)s; border-radius: 6px; }
#WorkspaceSubTool { background: %(raised)s; border: 1px solid %(line)s; border-radius: 7px; font-weight: 600; }
#WorkspaceSubTool:hover { background: %(raised_hi)s; border-color: %(accent)s; }
#WorkspaceTool, #WorkspaceToolPrimary { min-width: 76px; min-height: 58px; border-radius: 8px; font-weight: 600; }
#WorkspaceTool { border: 2px solid transparent; }
#RangeTestBox { background: %(well)s; border: 1px solid %(line)s; border-radius: 8px; }   /* same border width as the selected one, so picking a tool never shifts the row */
#WorkspaceFilmstrip { background: %(panel)s; border-top: 1px solid %(line)s; }
#WorkspaceRailToggle { background: #273444; border: 1px solid #53657a; border-radius: 9px; color: %(ink)s; font-size: 18px; font-weight: 700; }
#WorkspaceRailToggle:hover { background: #38516b; }
#WorkspaceFilmAction { background: %(accent)s; color: %(on_color)s; border: none; border-radius: 16px; min-width: 156px; font-weight: 700; padding: 6px 16px; }
#WorkspaceFilmAction:disabled { background: %(raised)s; color: %(ink3)s; }
#WorkspaceToolPrimary { color: %(accent_hi)s; border: 2px solid %(accent)s; background: %(raised)s; }
#WorkspaceTool:disabled { color: %(ink3)s; background: %(panel)s; border-color: %(line)s; }
#WorkspaceTool:checked { color: %(accent_hi)s; border: 2px solid %(accent)s; background: %(raised)s; }   /* Guide on: looks like the picked page */

/* ---- panels ---- */
QFrame[role="panel"] { background: %(panel)s; border: 1px solid %(line)s; border-radius: 10px; }
QLabel[role="section"] { color: %(ink2)s; font-weight: 600; }
QLabel[role="hint"] { color: %(ink3)s; }
QLabel[role="value"] { color: %(ink)s; font-weight: 600; }
QLabel[role="big"] { color: %(ink)s; font-weight: 700; }

/* ---- buttons ---- */
QPushButton { background: %(raised)s; border: 1px solid %(line)s; border-radius: 7px; padding: 6px 12px; }
QPushButton:hover { background: %(raised_hi)s; }
QPushButton:pressed { background: %(line)s; }
QPushButton:disabled { color: %(ink3)s; background: %(panel)s; }
QPushButton[kind="primary"] { background: %(kept)s; color: %(on_color)s; border: none; border-radius: 10px;
                              font-weight: 700; text-align: left; padding: 0 18px; }
QPushButton[kind="primary"]:hover { background: %(kept_hi)s; }
QPushButton[kind="pause"] { background: %(warn)s; color: %(on_color)s; border: none; border-radius: 10px;
                            font-weight: 700; text-align: left; padding: 0 18px; }
QPushButton[kind="pause"]:hover { background: %(warn_hi)s; }
QPushButton[kind="primary"]:disabled, QPushButton[kind="pause"]:disabled { background: %(raised)s; color: %(ink2)s; }
/* Build: tinted, not filled, so the big Start / Pause stays the one loud button */
QPushButton[kind="build"] { background: #1f3048; color: %(accent_hi)s; border: 1px solid #33527c; font-weight: 700; }
QPushButton[kind="build"]:hover { background: #274063; }
QPushButton[kind="build"]:disabled { background: %(panel)s; color: %(ink3)s; border-color: %(line)s; }
QPushButton[kind="quiet"] { background: transparent; border: 1px solid transparent; color: %(ink2)s; }
QPushButton[kind="quiet"]:hover { background: %(raised)s; color: %(ink)s; }
QPushButton[kind="quiet"]:disabled { color: %(ink3)s; background: transparent; }
QPushButton[kind="danger"] { background: transparent; border: 1px solid transparent; color: %(ink3)s; }
QPushButton[kind="danger"]:hover { color: %(lost)s; border-color: %(lost)s; }
QPushButton[kind="stop"] { background: #3b1c1b; color: %(lost)s; border: 1px solid #7a2e2a; font-weight: 700; }
QPushButton[kind="stop"]:hover { background: #4a2220; }
QLabel[role="keycap"] { border: 1px solid rgba(13,16,20,0.45); border-radius: 5px; padding: 1px 7px; }
QLabel[role="keycap-dark"] { background: %(raised)s; border: 1px solid %(line)s; border-bottom: 2px solid %(ink3)s;
                             border-radius: 5px; padding: 2px 8px; }
/* floating over the model */
QPushButton[kind="float"] { background: rgba(10,12,15,0.80); border: 1px solid %(line)s; border-radius: 7px; padding: 5px 12px; }
QPushButton[kind="float"]:hover { background: %(raised_hi)s; }
/* segmented switch: IR / Depth / Colour on short windows */
QPushButton[kind="choice"] { border-radius: 8px; padding: 6px 12px; color: %(ink2)s; background: %(raised)s; font-size: 15px; }
QPushButton[kind="choice"]:hover { background: %(raised_hi)s; color: %(ink)s; }
QPushButton[kind="choice"]:checked { background: %(raised_hi)s; color: %(ink)s; font-weight: 600; border: 2px solid %(accent)s; }
QPushButton[kind="choice"]:disabled { color: %(ink3)s; background: transparent; border-style: dashed; }
QPushButton[kind="seg"] { border-radius: 0; padding: 4px 10px; color: %(ink2)s; }
QPushButton[kind="seg"][pos="first"] { border-top-left-radius: 7px; border-bottom-left-radius: 7px; }
QPushButton[kind="seg"][pos="last"] { border-top-right-radius: 7px; border-bottom-right-radius: 7px; }
QPushButton[kind="seg"]:checked { background: %(raised_hi)s; color: %(ink)s; font-weight: 600; }
/* on/off chips: Auto gain, Auto range, Fill LED */
QPushButton[kind="chip"] { border-radius: 13px; padding: 3px 12px; color: %(ink2)s; background: %(raised)s; }
QPushButton[kind="chip"]:checked { background: #22382b; color: %(kept)s; border-color: #2f5a3b; }
QPushButton[kind="chip"]:disabled { color: %(ink3)s; background: transparent; border-style: dashed; }
QPushButton[kind="lamp"] { border-radius: 13px; padding: 3px 12px; color: %(ink2)s; background: %(raised)s; }
QPushButton[kind="lamp"]:checked { background: #3a3120; color: %(warn)s; border-color: #6a5528; }
QPushButton[kind="lamp"]:disabled { color: %(ink3)s; background: transparent; border-style: dashed; }

/* ---- sides ---- */
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QFrame[role="card"] { background: %(raised)s; border: 1px solid %(line)s; border-radius: 9px; }
QFrame[role="card"]:hover { border-color: %(ink3)s; }
QFrame[role="card"][picked="true"] { border: 2px solid %(accent)s; background: #262d38; }
QFrame[role="card"][live="true"] { border: 2px solid %(kept)s; background: #24302a; }
QLabel[role="badge-bed"] { color: %(platter)s; border: 1px solid #5b4386; border-radius: 8px; padding: 0 6px; }
QLabel[role="badge-auto"] { color: %(ink2)s; border: 1px solid %(line)s; border-radius: 8px; padding: 0 6px; }
QLabel[role="lost"] { color: %(lost)s; }
QLabel[role="live"] { color: %(kept)s; font-weight: 600; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: %(raised_hi)s; border-radius: 3px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: %(ink3)s; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: %(raised_hi)s; border-radius: 3px; min-width: 30px; }

/* ---- build and log ---- */
QProgressBar { background: %(well)s; border: 1px solid %(line)s; border-radius: 5px; height: 10px; text-align: center; }
QProgressBar::chunk { background: %(accent)s; border-radius: 4px; }
QPlainTextEdit#Log { background: %(well)s; border: 1px solid %(line)s; border-radius: 7px; color: %(ink2)s;
                     padding: 4px 6px; selection-background-color: %(accent)s; }
#LogPopover, #KeysSheet { background: %(panel)s; border: 1px solid %(line)s; border-radius: 12px; }

/* ---- status line and banner ---- */
#StatusLine { background: %(panel)s; border-top: 1px solid %(line)s; }
QLabel[role="stat"] { color: %(ink2)s; }
#Banner[level="warn"] { background: #3a2f17; border: 1px solid #6a5528; border-radius: 8px; }
#Banner[level="lost"] { background: #3b1c1b; border: 1px solid #7a2e2a; border-radius: 8px; }
#Banner[level="info"] { background: #1c2a3d; border: 1px solid #33527c; border-radius: 8px; }
#BannerText { font-weight: 600; }
""" % scan_style


def install(app):
    """Fonts and the style sheet for the whole app. Call once, before any window is shown; a second call is a no-op
    (re-setting the sheet re-polishes every live widget, which crawled in the test fixtures)."""
    if app.property("pointyoink_theme"): return
    app.setProperty("pointyoink_theme", True)
    QFontDatabase.families()       # makes Qt read fontconfig before the first font is asked for (offscreen too)
    app.setFont(ui_font(13))
    app.setStyle("Fusion")          # a plain base style, so the sheet looks the same on every desktop theme
    app.setStyleSheet(qss() + INPUT_QSS)
