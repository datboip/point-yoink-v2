import numpy as np
from PySide6.QtWidgets import QApplication

def test_what_is_left_uses_done_green_not_fine_gray():
    from app2.builder.scan_guide import quality_colors
    rgb=quality_colors(np.array([0,1,2],np.uint8))
    assert rgb.tolist()==[[70,215,120],[255,70,50],[180,90,230]]

def test_report_legend_matches_touchup_words():
    app=QApplication.instance() or QApplication([])
    from app2.qt.report_panel import ReportPanel
    panel=ReportPanel()
    text=' '.join(label.text() for label in panel.legend)
    for word in ['Not scanned yet (holes)','Rough or doubled, scan again','Done','#46d778']:
        assert word in text
    assert 'Fine' not in text
    panel.close()
