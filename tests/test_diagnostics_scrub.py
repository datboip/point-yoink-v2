"""The anonymizer: what it must hide, what it must keep, and that it is stable and idempotent."""
import re

from app2.diagnostics.scrub import Scrubber

CTX = dict(home="/home/alice", user="alice", host="testbox",
           roots=["/home/alice/revopoint-scans-models"],
           projects=["/home/alice/revopoint-scans-models/ProjectLive09282026192332",
                     "/home/alice/revopoint-scans-models/Project09222026020625"])

LINES = [
    "project folder: /home/alice/revopoint-scans-models/ProjectLive09282026192332",
    "bed: 29 dots (bed-20260927-181545, from the project, profile Bare turntable)",
    "$ systemd-run --user --scope -q -p MemoryMax=3G /home/alice/pointyoink/venv/bin/python /home/alice/pointyoink/scanner/stages/job.py /home/alice/revopoint-scans-models/ProjectLive09282026192332/stages/tmp/lineup.fix.5b6296aca576c950.1025931/job_1.json",
    "matched side 09282026222119 to side 09292026010758: point residuals [2.33, 1.41, 0.57] mm",
    "placed side 3 via side 1: 53% of it lands on the model so far (2 candidates through a table skipped)",
    "Wi-Fi share code 4821 from 192.168.77.10 (aa:bb:cc:dd:ee:ff)",
    "scanner serial: MIR123456789 firmware 1.0.0.129",
    "device: alice@testbox /dev/video0 is missing",
    "contact me at someone.else@example.com or https://example.org/report?token=abc123&user=alice",
    "peer fe80::1ff:fe23:4567:890a and 2001:db8::ff00:42:8329 answered",
    "opened C:\\Users\\Someone\\Documents\\scan.ply and /Users/sam/Desktop/x.ply and /home/other/y.ply",
    "key 8bf6ade5c4046ca6d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4",
    "/home/alice/revopoint-scans-models/Project09222026020625/data/09222026020625/cache/frame_000_0000.dph",
]

MUST_GO = ["alice", "testbox", "09282026192332", "09222026020625", "09282026222119", "09292026010758", "192.168.77.", "aa:bb:cc",
           "example.com", "example.org", "token=abc", "MIR123456789", "4821", "fe80:", "2001:db8", "Someone", "/Users/sam", "/home/other",
           "8bf6ade5c4046ca6d1e2"]

MUST_STAY = ["bed: 29 dots", "profile Bare turntable", "MemoryMax=3G", "residuals [2.33, 1.41, 0.57] mm", "53% of it lands",
             "2 candidates through a table skipped", "firmware 1.0.0.129", "/dev/video0 is missing", "frame_000_0000.dph", "lineup.fix"]


def test_nothing_that_identifies_survives_and_the_useful_words_do():
    s = Scrubber(CTX); out = "\n".join(s.scrub(l) for l in LINES)
    leaked = [m for m in MUST_GO if m in out]
    assert not leaked, (leaked, out)
    lost = [m for m in MUST_STAY if m not in out]
    assert not lost, (lost, out)
    assert not re.search(r"(?<!\d)\d{14}(?!\d)", out)                    # no raw scan time left anywhere


def test_the_same_input_gets_the_same_alias_and_different_inputs_differ():
    s = Scrubber(CTX)
    a = s.scrub("side 09282026222119 and side 09292026010758 and again side 09282026222119")
    assert a == "side side-1 and side side-2 and again side side-1"
    p = s.scrub("/home/alice/revopoint-scans-models/ProjectLive09282026192332 /home/alice/revopoint-scans-models/Project09222026020625 ProjectLive09282026192332")
    assert p == "Project-A Project-B Project-A"


def test_scrubbing_twice_changes_nothing_more():
    s = Scrubber(CTX)
    for line in LINES:
        once = s.scrub(line)
        assert Scrubber(CTX).scrub(once) == once


def test_versions_times_and_numbers_are_left_alone():
    s = Scrubber(CTX)
    keep = "PointYoink 2.0.0-pre.7 starting 2026-10-02 03:45:02,091 INFO OpenGL 3.3, NVIDIA GeForce RTX 3080/PCIe/SSE2 margin 2.8 mm 4756625 polygons"
    assert s.scrub(keep) == keep


def test_the_username_is_only_replaced_as_a_whole_word():
    s = Scrubber(dict(CTX, user="sam", host="pc"))
    assert s.scrub("sam opened samples and same") == "<user> opened samples and same"
    assert Scrubber(dict(CTX, user="ab", host="x")).scrub("ab") == "ab"           # too short to hide safely


def test_only_well_known_sites_stay_and_never_with_a_query():
    s = Scrubber(CTX)
    assert s.scrub("see https://github.com/datboip/point-yoink-v2/issues?x=1#top now") == "see https://github.com/datboip/point-yoink-v2/issues now"
    assert s.scrub("sent to https://internal.corp.example/path?a=b ok") == "sent to <url> ok"
