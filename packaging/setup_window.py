#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PointYoink first-run setup: shows what 2.0 needs, how big each part is, and downloads the picked parts into a private
folder (~/.local/share/pointyoink/venv). Runs on the system python3 with Tk (Qt is one of the parts it installs).

  python3 setup_window.py [--env DIR]      exit 0 when everything required is installed, 1 when cancelled or failed
"""
import argparse, json, os, queue, re, shutil, subprocess, sys, threading
import tkinter as tk
from tkinter import ttk

# Sizes are the downloads in MB (measured 2026-10-12); on disk it is about 2.9 times that. Keep the pins in step with
# packaging/requirements.txt.
PARTS = [
    dict(id="core", title="App window and basics", why="Qt for the window, plus numpy, scipy, mesh helpers and 3MF export", required=True,
         size=157, pkgs=["PySide6_Essentials==6.11.2", "numpy==2.4.2", "scipy==1.17.1", "PyOpenGL==3.1.10", "trimesh==5.1.0",
                         "pillow==12.3.0", "lz4==4.4.5", "threadpoolctl==3.6.0", "psutil==7.2.2", "fast_simplification==0.2.0",
                         "networkx==3.6.1", "lxml==5.2.1"]),
    dict(id="engine", title="3D engine (Open3D)", why="Fusion, line-up and the GPU work", required=True,
         choice={"gpu": dict(label="NVIDIA GPU build", note="faster, needed for live scanning", size=515, pkgs=["open3d==0.19.0"]),
                 "cpu": dict(label="CPU build", note="no NVIDIA card: builds work, live scanning does not", size=160,
                             pkgs=["open3d-cpu==0.19.0"])}),
    dict(id="mesh", title="Mesh tools (PyMeshLab)", why="Meshed, Finish for print, Cut shape, Add fin", required=True,
         size=107, pkgs=["pymeshlab==2025.7.post1"]),
    dict(id="camera", title="Camera and image tools (OpenCV)", why="Live camera views, bed dots, video thumbnails", required=True,
         size=62, pkgs=["opencv-python-headless==5.0.0.93"]),
]
INK, BG, PANEL, ACCENT, MUTED, BAD, OK = "#e6e9ee", "#16191e", "#1f242b", "#5aa0f2", "#8b95a1", "#e5534b", "#57c27a"


MSG_CHARS = 72                          # the progress line under a part keeps one width, so long file names never resize the window


def short(text, n=MSG_CHARS):
    return text if len(text) <= n else text[:n - 3] + "..."


def gb(mb): return "%d MB" % mb if mb < 1000 else "%.1f GB" % (mb / 1024)


# Open3D 0.19 has wheels for Python 3.8 to 3.12 only. On a newer system Python (SteamOS, Fedora 41+, Ubuntu 25.04+) the
# private folder gets its own Python 3.12 from uv (one small download, no admin password, kept inside that folder).
PY_OK = ((3, 10), (3, 12))
PY_PRIVATE = "3.12"
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-%s-unknown-linux-gnu.tar.gz"


def system_python_fits(v=None):
    v = tuple((v or sys.version_info)[:2]); return PY_OK[0] <= v <= PY_OK[1]


def fetch_uv(env, machine=None):
    """uv's single binary into <env>/uv/uv (downloaded once); returns its path."""
    import platform, tarfile, urllib.request
    dest = os.path.join(env, "uv"); exe = os.path.join(dest, "uv")
    if os.path.exists(exe): return exe
    os.makedirs(dest, exist_ok=True); tgz = os.path.join(dest, "uv.tar.gz.part")
    urllib.request.urlretrieve(UV_URL % (machine or platform.machine()), tgz)
    with tarfile.open(tgz) as t:
        m = next(m for m in t.getmembers() if m.isfile() and os.path.basename(m.name) == "uv")
        with t.extractfile(m) as src, open(exe + ".part", "wb") as out: shutil.copyfileobj(src, out)
    os.chmod(exe + ".part", 0o755); os.replace(exe + ".part", exe); os.remove(tgz)
    return exe


def private_python_cmds(env, uv):
    """The commands that make <env>/venv on uv's own Python (pip included, so the parts install the usual way)."""
    return [[uv, "python", "install", PY_PRIVATE], [uv, "venv", "--seed", "-q", "-p", PY_PRIVATE, os.path.join(env, "venv")]]


def uv_env(env):
    return dict(os.environ, UV_PYTHON_INSTALL_DIR=os.path.join(env, "python"), UV_CACHE_DIR=os.path.join(env, "uv", "cache"),
                UV_NO_MODIFY_PATH="1")


def has_nvidia():
    try: return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError): return False


def load_state(env):
    try:
        with open(os.path.join(env, "setup.json")) as f: return json.load(f)
    except (OSError, ValueError): return {}


def save_state(env, st):
    tmp = os.path.join(env, "setup.json.part")
    with open(tmp, "w") as f: json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(env, "setup.json"))


def ready(env):
    """Every required part installed (the launcher's check)."""
    st = load_state(env); have = set(st.get("parts", []))
    return os.path.exists(os.path.join(env, "venv", "bin", "python")) and all(p["id"] in have for p in PARTS if p["required"])


class Setup:
    def __init__(self, root, env):
        self.root, self.env = root, env; self.q = queue.Queue(); self.busy = False; self.ok = False
        self.state = load_state(env); have = set(self.state.get("parts", []))
        root.title("PointYoink setup"); root.configure(bg=BG); root.resizable(False, False)
        st = ttk.Style(root); st.theme_use("clam")
        st.configure(".", background=BG, foreground=INK, font=("Sans", 10))
        st.configure("TCheckbutton", background=PANEL); st.configure("TRadiobutton", background=PANEL)
        st.map("TCheckbutton", background=[("active", PANEL)]); st.map("TRadiobutton", background=[("active", PANEL)])
        st.configure("Horizontal.TProgressbar", troughcolor=BG, background=ACCENT, bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT)
        st.configure("TButton", background=PANEL, foreground=INK, padding=(14, 6)); st.map("TButton", background=[("active", "#2a313a")])
        st.configure("Go.TButton", background=ACCENT, foreground="#0d1117"); st.map("Go.TButton", background=[("active", "#7bb4f5"), ("disabled", PANEL)])

        f = tk.Frame(root, bg=BG, padx=22, pady=18); f.pack(fill="both")
        tk.Label(f, text="PointYoink setup", bg=BG, fg=INK, font=("Sans", 16, "bold")).pack(anchor="w")
        tk.Label(f, text="One time. These are the parts PointYoink 2.0 runs on. They download once into your own folder,\n"
                         "need no admin password, and go away with the app's data folder.", bg=BG, fg=MUTED, justify="left").pack(anchor="w", pady=(4, 12))
        gpu = has_nvidia(); self.engine = tk.StringVar(value=self.state.get("engine") or ("gpu" if gpu else "cpu"))
        self.rows = {}
        for p in PARTS:
            card = tk.Frame(f, bg=PANEL, padx=12, pady=9); card.pack(fill="x", pady=4)
            top = tk.Frame(card, bg=PANEL); top.pack(fill="x")
            var = tk.BooleanVar(value=p["required"] or p["id"] in have)
            cb = ttk.Checkbutton(top, text=p["title"], variable=var, command=self.refresh)
            if p["required"]: cb.state(["disabled"])
            cb.pack(side="left")
            size = tk.Label(top, bg=PANEL, fg=MUTED); size.pack(side="right")
            tag = "required" if p["required"] else "optional"
            if p["id"] in have: tag = "installed"
            tagl = tk.Label(top, text=tag, bg=PANEL, fg=OK if tag == "installed" else MUTED); tagl.pack(side="right", padx=10)
            tk.Label(card, text=p["why"], bg=PANEL, fg=MUTED).pack(anchor="w", padx=(22, 0))
            if p.get("choice"):
                for k, c in p["choice"].items():
                    t = "%s, %d MB: %s" % (c["label"], c["size"], c["note"]) + ("   (we found your NVIDIA card)" if k == "gpu" and gpu else "")
                    ttk.Radiobutton(card, text=t, value=k, variable=self.engine, command=self.refresh).pack(anchor="w", padx=(22, 0))
            bar = ttk.Progressbar(card, length=520, mode="determinate", maximum=100); msg = tk.Label(card, bg=PANEL, fg=MUTED, anchor="w", width=MSG_CHARS)
            self.rows[p["id"]] = dict(var=var, size=size, tag=tagl, bar=bar, msg=msg, card=card)
        bottom = tk.Frame(f, bg=BG); bottom.pack(fill="x", pady=(12, 0))
        self.where = tk.Label(bottom, bg=BG, fg=MUTED, anchor="w", justify="left"); self.where.pack(side="left")
        self.go = ttk.Button(bottom, text="Install", style="Go.TButton", command=self.start); self.go.pack(side="right")
        self.cancel = ttk.Button(bottom, text="Cancel", command=self.close); self.cancel.pack(side="right", padx=8)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh(); root.after(100, self.pump)

    def part_size(self, p):
        return p["choice"][self.engine.get()]["size"] if p.get("choice") else p["size"]

    def part_pkgs(self, p):
        return p["choice"][self.engine.get()]["pkgs"] if p.get("choice") else p["pkgs"]

    def wanted(self):
        have = set(self.state.get("parts", []))
        engine_changed = self.state.get("engine") and self.state.get("engine") != self.engine.get()
        return [p for p in PARTS if self.rows[p["id"]]["var"].get() and (p["id"] not in have or (p["id"] == "engine" and engine_changed))]

    def refresh(self):
        for p in PARTS: self.rows[p["id"]]["size"].config(text=gb(self.part_size(p)))
        todo = self.wanted(); mb = sum(self.part_size(p) for p in todo)
        where = self.env
        while not os.path.exists(where) and os.path.dirname(where) != where: where = os.path.dirname(where)      # a fresh account has no ~/.local/share yet
        free = shutil.disk_usage(where).free // 2**20
        self.where.config(text="Installs to %s\nDownload %s, about %s on disk. Free: %s" % (self.env, gb(mb), gb(mb * 2.9), gb(free)))
        if not self.busy: self.go.config(text="Install" if todo else "Start PointYoink")

    # ------------------------------------------------------------------ install
    def start(self):
        if self.busy: return
        todo = self.wanted()
        if not todo: self.finish(); return
        self.busy = True; self.go.state(["disabled"])
        for p in todo:
            r = self.rows[p["id"]]; r["bar"].config(value=0); r["bar"].pack(fill="x", pady=(6, 0)); r["msg"].pack(fill="x"); r["msg"].config(text="Waiting", fg=MUTED)
        threading.Thread(target=self.work, args=(todo, self.engine.get()), daemon=True).start()

    def work(self, todo, engine):
        env = self.env; py = os.path.join(env, "venv", "bin", "python")
        pip_env = dict(os.environ, PIP_CACHE_DIR=os.path.join(env, "pip-cache"))     # not ~/.cache/pip: removed once all parts are in
        try:
            os.makedirs(env, exist_ok=True)
            if not os.path.exists(py):
                if system_python_fits():
                    self.q.put(("msg", todo[0]["id"], "Making the private Python folder")); self.run([sys.executable, "-m", "venv", os.path.join(env, "venv")], None)
                else:
                    self.q.put(("msg", todo[0]["id"], "System Python is %d.%d; getting Python %s for PointYoink (35 MB)"
                                % (sys.version_info[0], sys.version_info[1], PY_PRIVATE)))
                    uv = fetch_uv(env)
                    for cmd in private_python_cmds(env, uv): self.run(cmd, None, env=uv_env(env))
                    shutil.rmtree(os.path.join(env, "uv", "cache"), ignore_errors=True)
                self.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], None, env=pip_env)
        except Exception as e:
            self.q.put(("fail", todo[0]["id"], str(e))); return
        for p in todo:
            pkgs = list(self.part_pkgs(p))
            if p["id"] == "engine": self.run([py, "-m", "pip", "uninstall", "-y", "-q", "open3d", "open3d-cpu"], None, check=False)
            self.q.put(("msg", p["id"], "Starting"))
            try: self.run([py, "-m", "pip", "install", "--progress-bar", "raw", "--disable-pip-version-check"] + pkgs, p, env=pip_env)
            except Exception as e:
                self.q.put(("fail", p["id"], str(e))); return
            st = load_state(env); parts = set(st.get("parts", [])); parts.add(p["id"]); st["parts"] = sorted(parts)
            if p["id"] == "engine": st["engine"] = engine
            save_state(env, st); self.state = st
            self.q.put(("done", p["id"], None))
        shutil.rmtree(os.path.join(env, "pip-cache"), ignore_errors=True)
        self.q.put(("all", None, None))

    def run(self, cmd, part, check=True, env=None):
        """pip with its raw progress ("Progress 123 of 456" per file) turned into this part's bar."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        total = (self.part_size(part) if part else 1) * 2**20; done = 0; cur = 0; tail = []
        for line in proc.stdout:
            line = line.strip(); tail = (tail + [line])[-6:]
            if part is None: continue
            m = re.match(r"Progress (\d+) of (\d+)", line)
            if m:
                cur = int(m.group(1))
                if cur >= int(m.group(2)): done += cur; cur = 0
                self.q.put(("bar", part["id"], min(99, 100 * (done + cur) / total)))
            elif line.startswith(("Collecting", "Downloading")):
                self.q.put(("msg", part["id"], short(line.split(" (")[0].replace("Collecting ", "Getting "))))
            elif line.startswith("Installing collected"):
                self.q.put(("msg", part["id"], "Installing (this part takes a minute)")); self.q.put(("spin", part["id"], None))
        proc.wait()
        if check and proc.returncode != 0:
            raise RuntimeError(next((t for t in reversed(tail) if t), "pip failed (exit %d)" % proc.returncode)[-160:])

    def pump(self):
        try:
            while True:
                kind, pid, val = self.q.get_nowait(); r = self.rows.get(pid)
                if kind == "bar": r["bar"].config(mode="determinate", value=val)
                elif kind == "spin": r["bar"].config(mode="indeterminate"); r["bar"].start(15)
                elif kind == "msg": r["msg"].config(text=short(val), fg=MUTED, wraplength=0)
                elif kind == "done":
                    r["bar"].stop(); r["bar"].config(mode="determinate", value=100); r["msg"].config(text="Installed", fg=OK); r["tag"].config(text="installed", fg=OK)
                elif kind == "fail":
                    r["bar"].stop(); self.busy = False; self.go.state(["!disabled"]); self.go.config(text="Retry")
                    r["msg"].config(text="Did not finish: %s. Check the internet connection, then Retry." % val, fg=BAD, wraplength=520, justify="left")
                elif kind == "all":
                    self.busy = False; self.go.state(["!disabled"]); self.refresh()
                    if ready(self.env): self.finish(); return
        except queue.Empty: pass
        self.root.after(100, self.pump)

    def finish(self):
        self.ok = ready(self.env)
        if self.ok: self.root.destroy()

    def close(self):
        if self.busy: return                      # pip half way: let it finish or fail (Retry picks up from there)
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "pointyoink"))
    a = ap.parse_args()
    root = tk.Tk(); s = Setup(root, a.env); root.mainloop()
    sys.exit(0 if s.ok else 1)


if __name__ == "__main__":
    main()
