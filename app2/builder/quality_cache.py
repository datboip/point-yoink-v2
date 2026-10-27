# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quality colors belong to both the surface and its placed frame set."""
import hashlib
import os


def identity(mesh, sides):
    digest = hashlib.sha256()
    for path in (mesh, sides):
        digest.update(os.path.realpath(path).encode())
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()[:20]
