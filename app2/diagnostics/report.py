# SPDX-License-Identifier: AGPL-3.0-or-later
"""Builds the text of a problem report from chosen sections. Pure Python, no Qt, no network.

`data` is gathered by the app (see shell._report_data); `sections` is the set of section keys the user left ticked; with
`anonymize` the whole text goes through the Scrubber. Nothing here reads scan data, meshes or images.
"""
from app2.diagnostics.scrub import Scrubber

# key, title, one-line description shown next to the checkbox
SECTIONS = (
    ("system", "App and computer", "Version, operating system, Python and Qt, the graphics card."),
    ("settings", "Settings in use", "Preset, cut method and margin, line-up and fusion settings. No file paths."),
    ("project", "Project summary", "How many sides and frames, how many are left out. Counts only."),
    ("popover", "What the Log window said", "The messages shown in the app's Log."),
    ("applog", "App log (last 300 lines)", "What the app recorded while it ran."),
    ("error", "The last error", "The most recent error with its trace."),
    ("steplog", "The failed step's log", "The end of the log of the build step that failed."),
    ("note", "My note", "What you typed in the box below."),
)
KEYS = tuple(k for k, _t, _d in SECTIONS)
WARNING = ("This is a copy of your report. Nothing has been sent and nothing is sent for you. Paste it into a text editor and read it "
           "before you share it anywhere. Anonymize removes the names and folders the app knows about, but it can miss things: names typed in notes, unusual file paths or hostnames, and unlabeled codes may remain.")
NEVER = "Never included: scan data, meshes, images, or the recipe with its paths."


def last_error(lines):
    """From the last error line of the app log to the end, at most 40 lines."""
    idx = [i for i, l in enumerate(lines) if "uncaught error" in l or " ERROR " in l or " CRITICAL " in l]
    return "\n".join(lines[idx[-1]:][:40]) if idx else ""


def system_lines(lines):
    """The version, system and graphics lines the log wrote at the start of the session."""
    return [l for l in lines if "PointYoink " in l and "starting" in l or " system: " in l or "model view:" in l][-4:]


def build_report(data, sections, anonymize=True, scrub_context=None, note=""):
    out = ["PointYoink problem report"]
    def add(key, title, body):
        if key in sections and body: out.append("\n== %s ==\n%s" % (title, body.rstrip()))
    titles = {k: t for k, t, _d in SECTIONS}
    add("system", titles["system"], "\n".join(data.get("system", [])))
    add("settings", titles["settings"], "\n".join("%s: %s" % (k, v) for k, v in (data.get("settings") or {}).items() if v not in (None, "")))
    add("project", titles["project"], "\n".join("%s: %s" % (k, v) for k, v in (data.get("project") or {}).items()))
    add("popover", titles["popover"], data.get("popover", ""))
    add("applog", titles["applog"], "\n".join((data.get("applog") or [])[-300:]))
    add("error", titles["error"], data.get("error", ""))
    add("steplog", titles["steplog"], data.get("steplog", ""))
    add("note", titles["note"], note.strip())
    text = "\n".join(out) + "\n"
    return Scrubber(scrub_context).scrub(text) if anonymize else text
