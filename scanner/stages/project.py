# SPDX-License-Identifier: AGPL-3.0-or-later
"""A live-scan project on disk: its sides (folders under data/), their ids and positions, leaf fingerprints, raw stats,
and the guard that keeps stages/ out of protected folders.

Side id = folder basename under data/ (e.g. 09272026002103). Position n (1..N over the sorted ids) is only a label and a
legacy file name, never a key: inserting or merging sides renumbers positions, ids stay."""
from pathlib import Path
import glob, json, os
from dataclasses import dataclass

from .keys import H, md5_file
from .keys import jload, jdump


class DeniedError(Exception): pass


@dataclass
class Side:
    id: str
    n: int
    folder: str
    frames: int
    tracked: int = None
    lost: int = None
    active: bool = True
    has_live_plane: bool = False


def denied(path):
    rp = os.path.realpath(path)
    for pre in (os.environ.get("POINTYOINK_STAGES_DENY") or "").split(":"):
        if not pre: continue
        pre = os.path.realpath(os.path.expanduser(pre))
        if rp == pre or rp.startswith(pre.rstrip("/") + "/"): return pre
    return None


class Project:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.data = os.path.join(self.root, "data"); self.build = os.path.join(self.root, "build")
        self.stages = os.path.join(self.root, "stages")
        self._sides = None; self._fp = {}; self._dir = {}; self.read_only = False; self.created = False

    @classmethod
    def open(cls, path, create=True, read_only=False):
        """create: make stages/ when missing. read_only: touch nothing on disk (status / show / history / list /
        versions); a project without stages/ then reads as a new one."""
        p = cls(path)
        if not os.path.isdir(p.data): raise FileNotFoundError("no data/ folder in %s" % p.root)
        if read_only:
            p.read_only = True; return p
        if create and not os.path.isdir(p.stages):
            d = denied(p.root)
            if d: raise DeniedError("refusing to create stages/ under %s (POINTYOINK_STAGES_DENY)" % d)
            p.created = True
            for sub in ("", "nodes", "cache", "runs", "tmp", "regions"): os.makedirs(os.path.join(p.stages, sub), exist_ok=True)
        else:
            p.created = False
            if os.path.isdir(p.stages):
                p.guard_write(p.stages)
                for sub in ("nodes", "cache", "runs", "tmp", "regions"): os.makedirs(os.path.join(p.stages, sub), exist_ok=True)
        return p

    def guard_write(self, path):
        """Refuse a write to path when it, or the folder it lands in, resolves (symlinks followed) under a
        POINTYOINK_STAGES_DENY prefix. Every publish target and legacy build/ file goes through here."""
        for q in (path, os.path.dirname(os.path.abspath(path))):
            d = denied(q)
            if d: raise DeniedError("refusing to write %s: it resolves to %s, under %s (POINTYOINK_STAGES_DENY)"
                                    % (path, os.path.realpath(q), d))

    def refresh(self):
        """Forget the leaf fingerprints: new frames of a side show up in the next plan (a long-lived engine calls this
        at the start of every plan / run / status)."""
        self._fp = {}

    # ------------------------------------------------------------------------------------------------------ sides
    def side_folders(self):
        return sorted(d for d in glob.glob(os.path.join(self.data, "*"))
                      if os.path.exists(os.path.join(d, "cache", "pointyoink_track.json")))

    def _scan(self, sid):
        """One directory scan of a side's cache (its .dph and .irl frames and the newest .dph time), kept while the directory's
        mtime stands (a frame added or removed changes it). The graph is built many times per status and each build asks for
        every side: 400 scans of 7,500-entry directories per status took 15 to 27 s once the dentries were cold."""
        d = os.path.join(self.folder(sid), "cache")
        try: mt = os.stat(d).st_mtime_ns
        except OSError: mt = None
        e = self._dir.get(sid)
        if e is not None and e["dir_mtime"] == mt: return e
        dph, irl, img, mx = [], {}, [], 0
        try:
            with os.scandir(d) as it:
                for x in it:
                    if x.name.endswith(".dph"):
                        dph.append(x.path)
                        try: mx = max(mx, x.stat().st_mtime_ns)
                        except OSError: pass
                    elif x.name.endswith(".img"): img.append(x.path)
                    elif x.name.endswith(".irl"):
                        try: irl[x.path] = x.stat()                    # the digest memo checks size, mtime and inode: no second stat per frame
                        except OSError: pass
        except OSError: pass
        e = {"dir_mtime": mt, "dph": sorted(dph), "irl": dict(sorted(irl.items())), "max_mtime": mx, "img": sorted(img)}
        self._dir[sid] = e; return e

    def sides(self, leave_out=(), with_tracked=False):
        out = []
        for i, d in enumerate(self.side_folders()):
            sid = os.path.basename(d)
            s = Side(sid, i + 1, d, len(self._scan(sid)["dph"]), active=sid not in leave_out,
                     has_live_plane=os.path.exists(os.path.join(d, "table_plane.json")))
            if with_tracked:
                st = self.raw_stats(sid); s.tracked, s.lost = st["tracked"], st["lost"]
            out.append(s)
        return out

    def folder(self, sid): return os.path.join(self.data, sid)

    def rel(self, path): return os.path.relpath(path, self.root)

    # ------------------------------------------------------------------------------------------ leaf fingerprints
    def frames_fp(self, sid):
        """H(id, count(*.dph), max mtime_ns(*.dph), md5(pose table), md5(Pl.bin)): build_project's side_key facts."""
        if sid in self._fp: return self._fp[sid]
        d = self.folder(sid); sc = self._scan(sid); dph = sc["dph"]; mt = sc["max_mtime"]
        fp = H([sid, len(dph), mt, md5_file(os.path.join(d, "cache", "pointyoink_register_pose.pose")),
                md5_file(os.path.join(d, "param", "Pl.bin"))])
        rgb=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in [Path(path) for path in sc["img"]]]
        fp=H([fp,rgb,md5_file(os.path.join(d,"param","Prgb.bin")),md5_file(os.path.join(d,"param","LC_RT.bin"))])
        self._fp[sid] = fp
        return fp

    def dph_stats(self, sid):
        sc = self._scan(sid)
        return len(sc["dph"]), sc["max_mtime"]

    def live_plane_fp(self, sid):
        return md5_file(os.path.join(self.folder(sid), "table_plane.json"))

    def board_frames_fp(self, sid):
        # IR changes must invalidate board recovery even when raw TSDF is unchanged. Once per side per refresh(): the graph
        # is built several times per status and each build asked for every side (36 memo loads per Side page switch).
        if ("board", sid) in self._fp: return self._fp[("board", sid)]
        from .keys import DigestMemo
        memo = DigestMemo(os.path.join(self.stages, "cache", "board-digests.json"), read_only=self.read_only)
        base = os.path.realpath(os.path.join(self.folder(sid), "cache"))      # real once, not once per frame (7,500 realpath walks)
        result = H([(os.path.basename(p), memo.digest(os.path.join(base, os.path.basename(p)), real=True, st=st)) for p, st in self._scan(sid)["irl"].items()])
        memo.save()
        self._fp[("board", sid)] = result
        return result

    def frame_size(self, sid):
        """(W, H) of the side's depth frames, like fuse.py: property.rvproj, else the file size."""
        d = self.folder(sid)
        try:
            sp = jload(os.path.join(d, "property.rvproj")).get("scan_param", {})
            w, h = int(sp.get("depth_width") or 0), int(sp.get("depth_height") or 0)
            if w and h: return [w, h]
        except Exception: pass
        dph = self._scan(sid)["dph"]
        if dph:
            px = os.path.getsize(dph[0]) // 2
            for w, h in ((800, 600), (400, 300), (640, 480), (1280, 800), (320, 240)):
                if w * h == px: return [w, h]
        return [800, 600]

    def raw_stats(self, sid):
        d = self.folder(sid); sc = self._scan(sid); n = len(sc["dph"])
        if "raw" in sc: return sc["raw"]                             # the pose table grows with the frames, so the scan's validity covers it
        pose = os.path.join(d, "cache", "pointyoink_register_pose.pose")
        tracked = 0
        if os.path.exists(pose):
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from fuse import read_global_poses
                tracked = len(read_global_poses(pose))
            except Exception: tracked = 0
        sc["raw"] = {"frames": n, "tracked": tracked, "lost": max(0, n - tracked),
                     "live_plane": os.path.exists(os.path.join(d, "table_plane.json"))}
        return sc["raw"]

    def resolve_side(self, token, sides=None):
        """A side given as an id or a 1-based position."""
        sides = sides or self.sides()
        for s in sides:
            if s.id == token: return s
        try:
            k = int(token)
            if 1 <= k <= len(sides) and len(str(token)) < 6: return sides[k - 1]
        except ValueError: pass
        raise ValueError("no side %r (give an id or a position 1..%d)" % (token, len(sides)))
