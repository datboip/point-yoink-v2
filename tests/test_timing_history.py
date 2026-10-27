"""Step timings are remembered across projects (a new project starts from real numbers) and the finish message says how long a run took."""
import json, os, tempfile
from pathlib import Path

os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink_timings_")) / "config.json")

from scanner.stages import store as S
from app2.builder.stagehost import took


class Prog:
    def __init__(self, secs, est): self._s = secs; self.plan = {"est_s": est}
    def elapsed(self): return self._s


def test_global_history_keeps_the_last_thirty_and_survives_a_missing_file():
    assert S.global_secs() == {}
    for k in range(35): S.note_global_secs("fused.tsdf", 100 + k)
    h = S.global_secs()["fused.tsdf"]
    assert len(h) == 30 and h[0] == 105.0 and h[-1] == 134.0


def test_took_shows_actual_and_estimate_and_nothing_for_a_cached_run():
    assert took(Prog(192, 240)) == " in 3:12 (est. 4 min)"
    assert took(Prog(1, 240)) == ""
    assert took(Prog(75, 0)) == " in 1:15"
    assert took(None) == ""
