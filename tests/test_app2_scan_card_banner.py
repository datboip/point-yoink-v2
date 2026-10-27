"""The build card and the banner over the scan view no longer sit on each other."""
import types
from app2.qt import scan_page as S


class Q:
    _card_top = S.ScanPage._card_top; _place_card = S.ScanPage._place_card
    def __init__(self, banner_visible, banner_h=70):
        self.banner = types.SimpleNamespace(isVisible=lambda: banner_visible, height=lambda: banner_h)
        self.moved = []
        self.build_card = types.SimpleNamespace(isVisible=lambda: True, width=lambda: 400, move=lambda x, y: self.moved.append((x, y)), raise_=lambda: None)
        self.model = types.SimpleNamespace(width=lambda: 900)


def test_card_starts_below_the_banner_and_goes_back_up_without_one():
    q = Q(True); q._place_card(); assert q.moved == [(250, 52 + 70 + 10)]
    q = Q(False); q._place_card(); assert q.moved == [(250, 52)]
    q = Q(True, banner_h=110); assert q._card_top() == 172


def test_a_hidden_card_is_left_alone():
    q = Q(True); q.build_card.isVisible = lambda: False; q._place_card(); assert q.moved == []
