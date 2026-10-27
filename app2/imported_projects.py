# SPDX-License-Identifier: AGPL-3.0-or-later
"""Identify scanner projects that have models but no PointYoink tracking."""
from pathlib import Path


RAW_ONLY_MESSAGE = ("This project only has raw scan frames. Choose Prepare on its card in Your projects to make "
                    "its sides available in Build.")


def imported_project(path):
    """Return (kind, model sides). Tracked PointYoink projects keep their normal Build flow."""
    data = Path(path) / "data"
    try:
        sides = sorted(p for p in data.iterdir() if p.is_dir() and not p.is_symlink())
    except OSError:
        return "empty", []
    if any((side / "cache" / "pointyoink_track.json").is_file() for side in sides):
        return "tracked", []

    models = []
    for side in sides:
        model = next((p for p in (side / "fuse_mesh.ply", side / "fuse.ply") if p.is_file() and not p.is_symlink()), None)
        if model is None:
            continue
        try:
            from stages import plyio
            faces = bool(plyio.counts(str(model)).get("face", 0))
        except (OSError, ValueError, KeyError):
            faces = False
        models.append(dict(name=side.name, path=str(model), preview=str(side / "preview.png"), faces=faces))
    if models:
        return "models", models
    if any(any(cache.glob("*.dph")) or any(cache.glob("*.inf")) for cache in (side / "cache" for side in sides) if cache.is_dir()):
        return "raw_only", []
    return "empty", []
