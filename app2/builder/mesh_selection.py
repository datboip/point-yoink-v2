# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mesh selection regions shared with the in-window editor."""
import numpy as np
from mesh_ops import FaceGraph


def magic_region(vertices, faces, seed, angle=49, visible=None, graph=None):
    """Grow the clicked face across edges within the seed normal's angle."""
    faces = np.asarray(faces)
    out = np.zeros(len(faces), bool)
    if not 0 <= int(seed) < len(faces): return out
    tri = np.asarray(vertices)[faces]
    normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths[:,None], 1e-12)
    allowed = (lengths > 1e-12) & (normals @ normals[seed] >= np.cos(np.deg2rad(angle)))
    if visible is not None: allowed &= np.asarray(visible, bool)
    allowed[seed] = True; out[seed] = True
    graph = graph if graph is not None else FaceGraph(faces)
    # Connected components avoid one whole-graph pass per face ring during hover.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = graph.pairs
    pairs = pairs[allowed[pairs[:,0]] & allowed[pairs[:,1]]]
    adjacency = coo_matrix((np.ones(len(pairs),np.uint8),(pairs[:,0],pairs[:,1])), shape=(len(faces),len(faces))).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    return allowed & (labels == labels[seed])
