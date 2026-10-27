# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export: the Build step's model (a mesh PLY) as STL, OBJ, PLY or 3MF, alone or in a ZIP with the project's side
folders (the scan data, for a backup or to share), plus the model's size in mm. Runs on the engine host's thread
(StageHost.h_export, h_model_dims), never on the GUI thread. STL is export_stl.py's writer; OBJ, PLY and 3MF go through
trimesh (a format trimesh cannot write here is left out of formats(), not offered and then failed). Vertex colors are
kept in PLY and OBJ when the model has them. Every file is written next to its target and renamed into place.

The last format picked is kept in export.json next to the app's config.json (POINTYOINK_CONFIG), like the saved build
presets are."""
import json, os, tempfile, zipfile
from pathlib import Path

import numpy as np

FORMATS = (("stl", "STL"), ("obj", "OBJ"), ("ply", "PLY"), ("3mf", "3MF"))
DEFAULT = "stl"
_OK = {}


def _can(fmt):
    """trimesh can write this format in this install (3MF needs its XML deps): tried once on a tiny mesh."""
    if fmt == "stl": return True
    if fmt not in _OK:
        try:
            import trimesh
            m = trimesh.Trimesh(np.eye(3), [[0, 1, 2]], process=False)
            _OK[fmt] = bool(m.export(file_type=fmt))
        except Exception: _OK[fmt] = False
    return _OK[fmt]


def formats():
    """[(key, label)] the dialog offers: STL always, the others when trimesh writes them."""
    return [(k, l) for k, l in FORMATS if _can(k)]


# ------------------------------------------------------------------------------------------------ size
def dims(V):
    """(width X, depth Y, height Z) of the axis-aligned box around the vertices, in mm (the file's units)."""
    V = np.asarray(V, np.float64)
    if not len(V): return (0.0, 0.0, 0.0)
    return tuple(float(x) for x in V.max(0) - V.min(0))


def dims_text(d):
    """"W x D x H mm", one decimal (a slicer's reading of the same file)."""
    return "%s x %s x %s mm" % tuple(("%.1f" % x) for x in d)


def ply_counts(path):
    """(vertices, faces, has_color) from a PLY's header alone (no data read)."""
    v = f = 0; color = False
    try:
        with open(path, "rb") as fh:
            for _ in range(80):
                line = fh.readline().decode("latin1", "replace").strip()
                if line.startswith("element vertex"): v = int(line.split()[2])
                elif line.startswith("element face"): f = int(line.split()[2])
                elif line.startswith("property") and line.endswith(" red"): color = True
                elif line == "end_header" or not line: break
    except (OSError, ValueError, IndexError): pass
    return v, f, color


def open_edges(path):
    """How many edges of the mesh at path belong to one triangle only (0 for a closed surface; None when it cannot be
    read or has no faces). Slicers count these too (Bambu Studio: "449 open edges", with a repair nag)."""
    import numpy as np
    try:
        from stages import plyio
        V, F, _N = plyio.read_mesh(path)
    except Exception: return None
    if F is None or not len(F): return None
    F = np.asarray(F, np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); E.sort(axis=1)
    keys = (E[:, 0].astype(np.uint64) << np.uint64(32)) | E[:, 1].astype(np.uint64)
    _u, cnt = np.unique(keys, return_counts=True)
    return int((cnt == 1).sum())


def size_estimates(v, f, color=False):
    """Rough file sizes in bytes per format from the counts. STL is exact (binary: 84 + 50 per triangle); the others are typical."""
    return {"stl": 84 + 50 * f,
            "obj": int(v * (30 + (24 if color else 0)) + f * 22),
            "ply": int(v * (12 + (3 if color else 0)) + f * 13 + 300),
            "3mf": int(0.3 * (v * 38 + f * 34))}


def human_bytes(n):
    n = float(n)
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div: return ("%.1f %s" if n < 10 * div else "%.0f %s") % (n / div, unit)
    return "%d bytes" % n


def folders_bytes(folders):
    """Total size of the files under these folders."""
    total = 0
    for top in folders:
        for root, _dirs, files in os.walk(top):
            for name in files:
                try: total += os.path.getsize(os.path.join(root, name))
                except OSError: pass
    return total


def model_dims(ply):
    """-> {"w", "d", "h", "text"} of a mesh PLY, reading its vertices only (no faces)."""
    from stages import plyio
    v = plyio.read(ply, faces=False)["vertex"]
    d = dims(plyio.xyz(v)) if v is not None else (0.0, 0.0, 0.0)
    return {"w": d[0], "d": d[1], "h": d[2], "text": dims_text(d)}


# ------------------------------------------------------------------------------------------------ write
def _read(ply):
    """-> (V, F, colors (n,3) uint8 or None)."""
    from stages import plyio
    d = plyio.read(ply); v = d["vertex"]
    F = d["faces"] if d["faces"] is not None else np.zeros((0, 3), np.int64)
    names = v.dtype.names or ()
    C = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8) if {"red", "green", "blue"} <= set(names) else None
    return plyio.xyz(v), np.asarray(F, np.int64), C


def _part(out):
    d = os.path.dirname(os.path.abspath(out)); os.makedirs(d, exist_ok=True)
    return os.path.join(d, ".%s.part" % os.path.basename(out))


def write_model(ply, out, fmt=DEFAULT):
    """-> {"out", "faces", "bytes", "dims", "format"}. Raises ValueError when the PLY has no triangles or the format is
    not one this install writes."""
    fmt = str(fmt).lower()
    if fmt not in dict(FORMATS) or not _can(fmt): raise ValueError("%s export is not available" % fmt.upper())
    V, F, C = _read(ply)
    if not len(F): raise ValueError("the model has no triangles")
    if fmt == "stl":
        from app2.builder.export_stl import write_stl
        r = write_stl(ply, out)
    elif fmt == "3mf" and C is not None:
        from app2.builder.color_3mf import write
        tmp = _part(out)
        try:
            write(tmp, V, F, C); os.replace(tmp, out)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
        r = {"out": out, "faces": int(len(F)), "bytes": os.path.getsize(out)}
    else:
        import trimesh
        m = trimesh.Trimesh(V, F, vertex_colors=None if C is None else np.c_[C, np.full(len(C), 255, np.uint8)], process=False)
        opts = {"include_texture": False} if fmt == "obj" else {}      # PLY and OBJ carry the vertex colors
        tmp = _part(out)
        try:
            data = m.export(file_type=fmt, **opts)
            with open(tmp, "wb") as f: f.write(data.encode() if isinstance(data, str) else data)
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
        r = {"out": out, "faces": int(len(F)), "bytes": os.path.getsize(out)}
    d = dims(V)
    r.update(dims=dims_text(d), format=fmt); return r


def side_folders(root):
    """The project's side folders (data/<side>), as the scanner saved them; an imported project with none of those
    still has its data/ folder."""
    from app2.scan import fmt as F
    sides = F.saved_sides(root)
    if sides: return sides
    data = os.path.join(root, "data")
    return [data] if os.path.isdir(data) and os.listdir(data) else []


def write_zip(ply, out, fmt, root):
    """The model (as fmt, named after the ZIP) plus every side folder under its path in the project -> out. -> the
    model's {"faces", "dims", ...} with "bytes" the ZIP's size and "sides" the side folders in it."""
    stem = os.path.splitext(os.path.basename(out))[0]
    tmp = _part(out)
    with tempfile.TemporaryDirectory(prefix="pointyoink-export-") as td:
        model = os.path.join(td, "%s.%s" % (stem, fmt))
        r = write_model(ply, model, fmt)
        sides = side_folders(root)
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
                z.write(model, os.path.basename(model))
                for s in sides:
                    for d, _dirs, files in os.walk(s):
                        for f in sorted(files):
                            p = os.path.join(d, f)
                            if os.path.isfile(p): z.write(p, os.path.relpath(p, root))
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
    r.update(out=out, bytes=os.path.getsize(out), sides=len(sides), zip=True); return r


def export(ply, out, fmt=DEFAULT, zip_root=None):
    """The one call the host makes: a model file, or (zip_root = the project folder) a ZIP with the scan data."""
    return write_zip(ply, out, fmt, zip_root) if zip_root else write_model(ply, out, fmt)


# ------------------------------------------------------------------------------------------------ last format
def prefs_path():
    config = Path(os.environ.get("POINTYOINK_CONFIG") or Path.home() / ".config/pointyoink/config.json")
    return config.parent / "export.json"


def last_format():
    try:
        f = json.loads(prefs_path().read_text()).get("format")
        return f if f in dict(FORMATS) else DEFAULT
    except (OSError, ValueError, AttributeError): return DEFAULT


def save_format(fmt):
    if fmt not in dict(FORMATS): return
    path = prefs_path(); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".export-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f: json.dump({"format": fmt}, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def orient_ply(ply, R):
    """A temporary copy of a mesh PLY turned by the 3x3 rotation R about the model's centre, then lowered so its bottom is at z = 0.
    -> path (the caller removes it). Used by Export for the orientation chosen on the Build page."""
    import numpy as np, tempfile
    from stages import plyio
    R = np.asarray(R, float).reshape(3, 3)
    V, F, N = plyio.read_mesh(ply); C = _read(ply)[2]; V = np.asarray(V, np.float64)
    c = V.mean(0); V = (V - c) @ R.T
    V[:, 2] -= V[:, 2].min()
    N = None if N is None else (np.asarray(N, np.float64) @ R.T).astype(np.float32)
    fd, tmp = tempfile.mkstemp(prefix="oriented-", suffix=".ply"); os.close(fd)
    plyio.write_mesh(tmp, V.astype(np.float32), F, N, colors=C)
    return tmp
