"""Where else a project's scan data lives, so a delete can say whether there is a backup."""
from app2.project_preview import backup_kind, BACKUP_WORDS


def test_live_scans_are_the_only_copy(tmp_path):
    for name in ("ProjectLive09282026192332", "ProjectLiveMerged-0926-0927"):
        assert backup_kind(tmp_path / name) == "none"


def test_scanner_projects_still_exist_on_the_scanner(tmp_path):
    assert backup_kind(tmp_path / "Project09222026020625") == "scanner"
    models = tmp_path / "Custom name"; models.mkdir(); (models / ".pointyoink-models-only").write_text("")
    assert backup_kind(models) == "scanner"


def test_anything_else_is_unknown_and_says_nothing(tmp_path):
    assert backup_kind(tmp_path / "My scan") == "unknown"
    assert BACKUP_WORDS["unknown"] == ("", "")
    assert "no backup" in BACKUP_WORDS["none"][1].lower()
