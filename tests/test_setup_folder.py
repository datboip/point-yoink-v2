import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging'))
import setup_window as sw


def test_a_chosen_folder_is_remembered_and_the_default_forgets_it(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    default = str(tmp_path / 'share' / 'pointyoink'); other = str(tmp_path / 'big' / 'pointyoink')
    sw.remember_env(other, default)
    assert Path(sw.pointer_file()).read_text().strip() == other
    sw.remember_env(default, default)
    assert not os.path.exists(sw.pointer_file())


def test_the_launcher_follows_the_pointer_file(tmp_path):
    import re, subprocess
    text = Path(__file__).resolve().parents[1].joinpath('packaging/build-deb2.sh').read_text()
    block = '\n'.join(l for l in text.splitlines() if re.match(r'(ENV=|PTR=|\[ -s "\$PTR")', l)) + '\n'
    cfg = tmp_path / 'cfg' / 'pointyoink'; cfg.mkdir(parents=True); (cfg / 'install_dir').write_text('/elsewhere/pointyoink\n')
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / 'cfg'), HOME=str(tmp_path))
    out = subprocess.run(['sh', '-c', block + 'echo "$ENV"'], env=env, capture_output=True, text=True).stdout.strip()
    assert out == '/elsewhere/pointyoink'
