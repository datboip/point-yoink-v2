# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode the stage worker event protocol."""
import json

def _stage_json(line, name):
    if line.startswith("STAGE %s " % name):
        try: return json.loads(line.split(" ", 2)[2])
        except ValueError: return None
    return None


