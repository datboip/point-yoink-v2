"""Regressions that also run unchanged against the held-layout baseline."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen');os.environ['POINTYOINK_NO_DEVICE']='1'
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from app2.qt.scan_page import ScanPage
from app2.qt.bridge import SessionBridge
from app2.scan.range_guide import describe
from test_app2_workflow_fixes import state


@pytest.fixture
def page(tmp_path):
    app=QApplication.instance() or QApplication([]);p=ScanPage(SessionBridge(),False);p.resize(1280,800);p.show();p._settings_user_choice=True
    p.on_state(state(str(tmp_path),scan_mode='near',range_guide=describe(np.full((80,100),3000,np.uint16),(100,300))))
    p.relayout(p.width(),p.height())
    for _ in range(6):app.processEvents()
    yield p
    p.close();app.processEvents()


def test_object_choice_has_job_names_and_compatible_keys(page):
    c=page.controls.scan_mode
    assert [(c.itemText(i),c.itemData(i)) for i in range(2)]==[('Small part','near'),('Large part','far')]


def test_idle_progress_collapses_to_the_build_button(page):
    box=page.sides.buildbox
    assert box.height()==box.build.height()


def test_footer_does_not_cover_the_image(page):
    bottom=page.model.image.mapTo(page,QPoint(0,page.model.image.height())).y()
    assert bottom<=page.distance_bar.y()


def test_calibration_folder_picker_is_out_of_the_main_settings(page):
    assert not page.controls.choose_far.isVisible()
