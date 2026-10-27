# Real-worker saved-frame fixture

This fixture contains eight depth frames from a saved one-side scan. Frames were selected every 62 saved frames, spanning both scan passes. The archive contains only their original depth and per-frame pose files, the matching rows of both registered pose tables, the matching tracking stamps, and depth calibration. It contains no built models or cache results.

Archive: 358,229 bytes. Expanded files: 1,926,007 bytes in 20 files. CPU smoke settings: 2 mm voxels, eight frames, no GPU, Meshed as Fused. The test extracts into pytest temporary storage, builds with real workers, deletes a small face selection, checks exact Undo and Redo outputs, then closes and reopens the private project host.

Run: `python -m pytest -s -q tests/test_real_worker_smoke.py`.
Select the marker with `-m real_worker`; exclude it with `-m "not real_worker"`. It skips only when the listed Python dependencies are missing. The full suite on a development machine must run it.
