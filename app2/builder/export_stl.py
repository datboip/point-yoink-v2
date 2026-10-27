# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export STL: a mesh PLY (the Build step's model) as a binary STL, written next to its target and renamed into place
(a half-written STL never sits under the chosen name). Binary STL = an 80-byte header, a uint32 count, then 50 bytes a
triangle (normal, three corners, a 2-byte attribute): 84 + 50 x triangles bytes, the size the Build step estimates."""
import os

import numpy as np

HEADER = b"PointYoink binary STL"


def stl_size(faces):
    return 84 + 50 * int(faces)


def write_stl(ply, out):
    """-> {"out", "faces", "bytes"}. Raises ValueError when the PLY has no triangles."""
    from stages import plyio
    V, F, _N = plyio.read_mesh(ply)
    V = np.asarray(V, np.float32); F = np.asarray(F, np.int64)
    if not len(F): raise ValueError("the model has no triangles")
    tri = V[F]                                                      # (n, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); n = np.where(ln > 0, n / np.where(ln > 0, ln, 1), 0).astype(np.float32)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n; rec["v"] = tri
    d = os.path.dirname(os.path.abspath(out)); os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".%s.part" % os.path.basename(out))
    try:
        with open(tmp, "wb") as f:
            f.write(HEADER.ljust(80, b" ")); f.write(np.uint32(len(F)).tobytes()); f.write(rec.tobytes())
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return {"out": out, "faces": int(len(F)), "bytes": os.path.getsize(out)}
