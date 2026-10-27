#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess entry for in-engine work: python job.py <job.json>
job.json carries "fn" ("module.path:function" under stages/), "out_dir", "result" and the job's own fields. The
function's return value is written to job["result"]. Runs under the runner's memory cap like any other step."""
import importlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.dirname(HERE)
REPO = os.path.dirname(SCANNER)
if sys.path and os.path.abspath(sys.path[0]) == HERE: sys.path.pop(0)
for p in (REPO, SCANNER):
    if p not in sys.path: sys.path.insert(0, p)


def main():
    with open(sys.argv[1]) as f: job = json.load(f)
    mod, fn = job["fn"].split(":")
    m = importlib.import_module("stages." + mod)
    res = getattr(m, fn)(job)
    if job.get("result"):
        tmp = job["result"] + ".tmp"
        with open(tmp, "w") as f: json.dump(res if res is not None else {}, f)
        os.replace(tmp, job["result"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
