"""Stable applied edits and Finish for print during status refreshes."""
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt.tool_panel import AppliedList
from app2.qt.stage_panels import StagePanel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def row(**values):
    return dict(id="finish", on=True, label="Finish for print", recipe_key="v1", **values)


def test_repeated_status_keeps_applied_widgets(app):
    panel = AppliedList()
    panel.set_rows("Meshed", [row()])
    original = panel.rows[0][0]
    for _ in range(50):
        panel.set_rows("Meshed", [row()])
        assert panel.rows[0][0] is original


def test_status_preserves_pending_checks(app):
    panel = AppliedList()
    panel.set_rows("Meshed", [row()])
    checkbox = panel.checks["finish"]
    checkbox.setChecked(False)
    panel.set_rows("Meshed", [row()])
    assert panel.checks["finish"] is checkbox
    assert panel.pending == {"finish": False}


def test_changed_cache_and_editing_are_rendered(app):
    panel = AppliedList()
    panel.set_rows("Meshed", [row(cached=False)])
    old = panel.rows[0][0]
    panel.set_rows("Meshed", [row(cached=True, editing=True)])
    assert panel.rows[0][0] is not old
    assert panel.badges["finish"].text() == "Cached"


def test_changed_notes_and_sections_are_rendered(app):
    panel = AppliedList()
    panel.set_rows("Meshed", [row()])
    old = panel.rows[0][0]
    panel.set_rows("Meshed", [row()], ["Saved result"], [(0, "Surface edits")])
    assert panel.rows[0][0] is not old
    assert panel.body.itemAt(0).widget().text() == "Surface edits"


def test_as_scanned_never_hides_basic_poisson_controls(app):
    panel = StagePanel("meshed")
    controls = [c for c in panel.bodies["Mesh"].controls
                if c.c.get("show_if") == "poisson" and not c.c.get("advanced")]
    assert controls
    panel.show_if({"mesh", "as_fused"})
    for control in controls:
        assert not control.isHidden()
        assert control.property("shown") is False
