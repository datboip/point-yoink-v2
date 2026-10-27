# SPDX-License-Identifier: AGPL-3.0-or-later
"""PLY and points.npz without Open3D (numpy only), so the engine process stays light and pml_mesh.py can share it.

points.npz (fused points and point-tool outputs): P float32 (n,3) mm, N float32 (n,3) unit normals, W float32 weight,
NF float32 frames that saw the point, S uint64 side bitmask (bit k = side_ids[k]), side_ids str array, cell float.
points.ply (only when a PLY consumer needs it): x y z nx ny nz quality[, red green blue], binary little endian,
quality = log1p(W)."""
import os

import numpy as np

_T = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
      "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_header(path):
    """-> (format, [(element name, count, [(prop, dtype) or (prop, 'list', count dtype, item dtype)])], data offset)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply": raise ValueError("%s is not a PLY file" % path)
        fmt = None; els = []
        while True:
            ln = f.readline()
            if not ln: raise ValueError("%s: truncated header" % path)
            t = ln.decode("ascii", "replace").split()
            if not t: continue
            if t[0] == "format": fmt = t[1]
            elif t[0] == "element": els.append([t[1], int(t[2]), []])
            elif t[0] == "property":
                if t[1] == "list": els[-1][2].append((t[4], "list", _T[t[2]], _T[t[3]]))
                else: els[-1][2].append((t[2], _T[t[1]]))
            elif t[0] == "end_header": return fmt, els, f.tell()


def counts(path):
    """Header-only element counts: {'vertex': n, 'face': m}."""
    _, els, _ = read_header(path)
    return {e[0]: e[1] for e in els}


def read(path, faces=True):
    """-> dict(vertex=structured array, faces=(m,3) int array or None)."""
    fmt, els, off = read_header(path)
    out = {"vertex": None, "faces": None}
    if fmt == "ascii":
        return _read_ascii(path, els, off, faces)
    end = "<" if fmt == "binary_little_endian" else ">"
    with open(path, "rb") as f:
        f.seek(off); buf = f.read()
    pos = 0
    for name, n, props in els:
        if all(len(p) == 2 for p in props):
            dt = np.dtype([(p[0], end + p[1]) for p in props])
            arr = np.frombuffer(buf, dt, count=n, offset=pos); pos += n * dt.itemsize
            if name == "vertex": out["vertex"] = arr
            continue
        if name == "face" and len(props) == 1:
            _, _, ct, it = props[0]
            cdt = np.dtype(end + ct); idt = np.dtype(end + it)
            # the usual case: every face a triangle
            rec = np.dtype([("c", cdt), ("i", idt, (3,))])
            if n and np.frombuffer(buf, cdt, count=1, offset=pos)[0] == 3:
                arr = np.frombuffer(buf, rec, count=n, offset=pos)
                if (arr["c"] == 3).all():
                    pos += n * rec.itemsize
                    if faces: out["faces"] = arr["i"].astype(np.int64)
                    continue
            F = []
            for _ in range(n):
                c = int(np.frombuffer(buf, cdt, count=1, offset=pos)[0]); pos += cdt.itemsize
                ids = np.frombuffer(buf, idt, count=c, offset=pos); pos += c * idt.itemsize
                for k in range(1, c - 1): F.append((ids[0], ids[k], ids[k + 1]))
            if faces: out["faces"] = np.array(F, np.int64).reshape(-1, 3)
            continue
        raise ValueError("%s: element %s with list properties is not supported" % (path, name))
    return out


def _read_ascii(path, els, off, faces):
    with open(path, "rb") as f:
        f.seek(off); lines = f.read().decode().split("\n")
    out = {"vertex": None, "faces": None}; k = 0
    for name, n, props in els:
        rows = lines[k:k + n]; k += n
        if name == "vertex":
            A = np.array([[float(x) for x in r.split()[:len(props)]] for r in rows])
            dt = np.dtype([(p[0], "<f8") for p in props]); arr = np.zeros(n, dt)
            for j, p in enumerate(props): arr[p[0]] = A[:, j]
            out["vertex"] = arr
        elif name == "face" and faces:
            F = []
            for r in rows:
                t = [int(x) for x in r.split()]
                for q in range(1, t[0] - 1): F.append((t[1], t[1 + q], t[2 + q]))
            out["faces"] = np.array(F, np.int64).reshape(-1, 3)
    return out


def xyz(v):
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def normals(v):
    if v.dtype.names and "nx" in v.dtype.names: return np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64)
    return None


def read_mesh(path):
    """-> (V (n,3) float64, F (m,3) int64 or empty, N (n,3) or None)."""
    d = read(path)
    F = d["faces"] if d["faces"] is not None else np.zeros((0, 3), np.int64)
    return xyz(d["vertex"]), F, normals(d["vertex"])


def vertex_mean(path):
    return xyz(read(path, faces=False)["vertex"]).mean(0)


def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])      # area weighted
    N = np.zeros_like(V)
    for k in range(3): np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def write_mesh(path, V, F, N=None, colors=None):
    """Binary little-endian PLY: float32 xyz (+ normals, + uchar colours), uchar/int face lists."""
    V = np.asarray(V); F = np.asarray(F, np.int64).reshape(-1, 3)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if N is not None: fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colors is not None: fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(len(V), dtype=fields)
    arr["x"], arr["y"], arr["z"] = V[:, 0], V[:, 1], V[:, 2]
    if N is not None: arr["nx"], arr["ny"], arr["nz"] = N[:, 0], N[:, 1], N[:, 2]
    if colors is not None:
        c = np.asarray(colors)
        if c.dtype != np.uint8: c = np.clip(np.round(c * 255 if c.max() <= 1.0 else c), 0, 255).astype(np.uint8)
        arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    h = ["ply", "format binary_little_endian 1.0", "element vertex %d" % len(V)]
    h += ["property %s %s" % ({"<f4": "float", "u1": "uchar"}[t], n) for n, t in fields]
    h += ["element face %d" % len(F), "property list uchar int vertex_indices", "end_header"]
    fa = np.empty(len(F), dtype=[("c", "u1"), ("i", "<i4", (3,))]); fa["c"] = 3; fa["i"] = F
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(("\n".join(h) + "\n").encode()); f.write(arr.tobytes()); f.write(fa.tobytes())
    os.replace(tmp, path)


def write_points_ply(path, P, N, Q=None, colors=None):
    n = len(P)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if Q is not None: fields.append(("quality", "<f4"))
    if colors is not None: fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = np.asarray(P, np.float32).T; arr["nx"], arr["ny"], arr["nz"] = np.asarray(N, np.float32).T
    if Q is not None: arr["quality"] = np.asarray(Q, np.float32)
    if colors is not None:
        c = np.asarray(colors)
        if c.dtype != np.uint8: c = np.clip(np.round(c * 255 if c.max() <= 1.0 else c), 0, 255).astype(np.uint8)
        arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    h = ["ply", "format binary_little_endian 1.0", "element vertex %d" % n]
    h += ["property %s %s" % ({"<f4": "float", "u1": "uchar"}[t], nm) for nm, t in fields]
    h.append("end_header")
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(("\n".join(h) + "\n").encode()); f.write(arr.tobytes())
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------ geometry digest
GEO_Q = 1e-5            # mm: vertex positions are rounded to 0.01 um before hashing (float32 values that are equal stay equal)


def geo_digest_arrays(V, F):
    """Order-independent digest of a triangle mesh: the same surface written with its vertices and faces in another
    order (the GPU TSDF, Poisson's threads) gives the same digest. Vertices are rounded (GEO_Q) and deduplicated, each
    triangle is written as position ids rotated so the smallest comes first (orientation kept), and the sorted triangles
    and sorted positions are hashed with the counts. Normals are left out: here they follow from the surface."""
    import hashlib
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64).reshape(-1, 3)
    q = np.round(V / GEO_Q).astype(np.int64)
    uq, inv = np.unique(q, axis=0, return_inverse=True) if len(q) else (np.zeros((0, 3), np.int64), np.zeros(0, np.int64))
    inv = np.asarray(inv).reshape(-1)
    T = inv[F] if len(F) else np.zeros((0, 3), np.int64)
    if len(T):
        r = np.argmin(T, axis=1); k = np.arange(len(T))
        T = np.stack([T[k, r], T[k, (r + 1) % 3], T[k, (r + 2) % 3]], 1)
        T = T[np.lexsort((T[:, 2], T[:, 1], T[:, 0]))]
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array([len(V), len(F), len(uq)], np.int64).tobytes())
    h.update(np.ascontiguousarray(uq).tobytes()); h.update(np.ascontiguousarray(T).tobytes())
    return "g2:" + h.hexdigest()


def geo_digest(path, max_faces=None):
    """geo_digest_arrays of a mesh PLY, or None when it is not a triangle mesh (points) or has more than max_faces."""
    c = counts(path)
    if not c.get("face"): return None
    if max_faces is not None and c["face"] > max_faces: return None
    V, F, _ = read_mesh(path)
    return geo_digest_arrays(V, F)


def geo_digest_job(job):
    """engine job (under a memory cap) for meshes too big to digest in the engine process."""
    return {"geo": geo_digest(job["in"])}


# ------------------------------------------------------------------------------------------------------ points.npz
def save_points(path, P, N, W, NF, S, side_ids, cell, colors=None):
    tmp = path + ".tmp.%d.npz" % os.getpid()
    np.savez(tmp, P=np.asarray(P, np.float32), N=np.asarray(N, np.float32), W=np.asarray(W, np.float32),
             NF=np.asarray(NF, np.float32), S=np.asarray(S, np.uint64), side_ids=np.asarray(side_ids, dtype=str),
             cell=np.float64(cell), **({"C":np.asarray(colors,np.uint8)} if colors is not None else {}))
    os.replace(tmp, path)


def load_points(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def points_count(path):
    """Point count of a points.npz without loading it (reads the P array header)."""
    import zipfile
    with zipfile.ZipFile(path) as zf:
        with zf.open("P.npy") as f:
            v = np.lib.format.read_magic(f)
            shape = (np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f))[0]
    return int(shape[0])


def load_any(path):
    """-> ("points", dict) for a points.npz, ("mesh", (V, F, N)) for a PLY."""
    if path.endswith(".npz"): return "points", load_points(path)
    return "mesh", read_mesh(path)
