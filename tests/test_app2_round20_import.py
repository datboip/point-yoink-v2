"""Import connection choices remain readable when the shell is short."""
import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


@pytest.mark.parametrize('width,height', [(1280,800), (1366,768), (1920,1080)])
def test_import_choices_keep_labels_readable_and_destination_reachable(monkeypatch, tmp_path, width, height):
    from app2.qt import theme, device_projects as module
    monkeypatch.setenv('POINTYOINK_NO_DEVICE', '1')
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setattr(module, 'usb_mode', lambda: None)
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(0)
    header = QWidget()
    header.setFixedHeight(168)
    layout.addWidget(header)
    page = module.DeviceProjects(str(tmp_path), window)
    layout.addWidget(page)
    window.setFixedSize(width,height)
    window.show()
    page.show()
    QTest.qWait(80)
    try:
        for card in page.way_b.values():
            labels = [label for label in card.findChildren(QLabel) if label.text()]
            rectangles = [QRect(label.mapTo(card,QPoint()),label.size()) for label in labels]
            assert all(card.rect().contains(rect) for rect in rectangles)
            assert all(label.height() >= label.fontMetrics().height() for label in labels)
            assert not any(a.intersects(b) for i,a in enumerate(rectangles) for b in rectangles[i+1:]), [(label.text(), rect.getRect()) for label,rect in zip(labels,rectangles)]
        scroll = getattr(page, 'connection_scroll', None)
        if scroll is not None:
            assert scroll.horizontalScrollBar().maximum() == 0
            scroll.ensureWidgetVisible(page.change_b)
        QTest.qWait(30)
        viewport = scroll.viewport() if scroll is not None else page.connection
        rect = QRect(page.change_b.mapTo(viewport,QPoint()),page.change_b.size())
        assert viewport.rect().contains(rect)
        if scroll is not None:
            scroll.ensureWidgetVisible(page.connect_b)
            QTest.qWait(30)
        # The scroll body must not override the primary button's blue background.
        colour = page.connect_b.grab().toImage().pixelColor(8, 8)
        assert (colour.red(), colour.green(), colour.blue()) == (91, 157, 240)
        assert page.proc is None
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
