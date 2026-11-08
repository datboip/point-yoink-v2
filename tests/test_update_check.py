# SPDX-License-Identifier: AGPL-3.0-or-later
from app2 import update_check as u


def test_parse_and_compare():
    assert u.parse('v2.0.4') == (2, 0, 4) and u.parse('2.0.10') == (2, 0, 10) and u.parse('2.0.4-pre') is None
    assert u.compare('2.0.3', 'v2.0.4') == 'newer'
    assert u.compare('2.0.9', 'v2.0.10') == 'newer'
    assert u.compare('2.0.4', 'v2.0.4') == 'current' and u.compare('2.1.0', 'v2.0.9') == 'current'
    assert u.compare('2.0.3', '') == 'unknown' and u.compare('2.0.3', 'nightly') == 'unknown'


def test_latest_survives_a_dead_network():
    assert u.latest(lambda: 'v2.0.4') == 'v2.0.4'
    def boom(): raise OSError('offline')
    assert u.latest(boom) == ''
