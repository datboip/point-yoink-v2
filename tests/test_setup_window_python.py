"""Setup window: a system Python Open3D has no wheels for gets a private Python 3.12 from uv, kept inside the app's folder;
progress lines keep one width."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packaging"))
import pytest
tk = pytest.importorskip("tkinter")
import setup_window as S


def test_which_system_pythons_fit():
    assert S.system_python_fits((3, 12)) and S.system_python_fits((3, 10))
    assert not S.system_python_fits((3, 13)) and not S.system_python_fits((3, 9))


def test_private_python_stays_in_the_app_folder(tmp_path):
    env = str(tmp_path / "pointyoink"); uv = os.path.join(env, "uv", "uv")
    install, venv = S.private_python_cmds(env, uv)
    assert install == [uv, "python", "install", S.PY_PRIVATE]
    assert venv[:2] == [uv, "venv"] and "--seed" in venv and venv[-1] == os.path.join(env, "venv")
    e = S.uv_env(env)
    assert e["UV_PYTHON_INSTALL_DIR"].startswith(env) and e["UV_CACHE_DIR"].startswith(env) and e["UV_NO_MODIFY_PATH"] == "1"


def test_progress_text_keeps_one_width():
    long = "Downloading scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    assert len(S.short(long)) == S.MSG_CHARS and S.short(long).endswith("...") and S.short("Waiting") == "Waiting"
