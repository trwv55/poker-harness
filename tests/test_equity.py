from harness.analysis.tools.equity import equity_hand_vs_hand, equity_vs_range
from harness.contracts import Range


def test_anchor_aks_vs_qq():
    assert abs(equity_hand_vs_hand(("As", "Ks"), ("Qh", "Qd")) - 0.46) < 0.015


def test_anchor_aa_vs_kk():
    assert abs(equity_hand_vs_hand(("Ah", "Ad"), ("Kh", "Kd")) - 0.815) < 0.02


def test_equity_vs_range_monotone():
    wide = Range(weights={c: 1.0 for c in ["22", "33", "A2s", "K9o", "QTs", "76s"]})
    tight = Range(weights={"AA": 1.0, "KK": 1.0})
    e_wide = equity_vs_range(("Qs", "Qh"), wide)
    e_tight = equity_vs_range(("Qs", "Qh"), tight)
    assert e_wide > 0.6 > e_tight  # QQ впереди широкого, позади AA/KK
    assert e_tight < 0.25


def test_equity_deterministic_with_seed():
    r = Range(weights={"AKo": 1.0, "AKs": 1.0})
    assert equity_vs_range(("Th", "Td"), r) == equity_vs_range(("Th", "Td"), r)


def test_multiway_equity_below_headsup():
    from harness.analysis.tools.equity import equity_vs_ranges

    r = Range(weights={"AKo": 1.0, "AKs": 1.0})
    hu = equity_vs_ranges(("Qs", "Qh"), [r])
    three = equity_vs_ranges(("Qs", "Qh"), [r, r])
    assert hu > three  # против двух AK доля банка меньше, чем против одного
    assert 0.35 < three < 0.55  # QQ против двух AK — примерно паритет
