from harness.analysis.tools.equity import equity_hand_vs_hand, equity_vs_range
from harness.contracts import Range, all_classes


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


def test_multiway_equity_drops_against_independent_ranges():
    # контроль: со случайными руками блокеры размыты, и доля банка падает,
    # как и ожидается от лишнего оппонента. Проверено перебором: 0.8005 -> 0.6498
    from harness.analysis.tools.equity import equity_vs_ranges

    any_two = Range(weights={c: 1.0 for c in all_classes()})
    one = equity_vs_ranges(("Qs", "Qh"), [any_two])
    two = equity_vs_ranges(("Qs", "Qh"), [any_two, any_two])
    assert two < one - 0.05


def test_multiway_blockers_can_raise_equity():
    # НЕ опечатка: против ДВУХ оппонентов с одинаковым узким AK доля QQ ВЫШЕ,
    # чем против одного. Два AK съедают тузов и королей друг друга: вероятность
    # борда без туза и короля растёт с 0.4968 до 0.6206. Проверено точным
    # перебором всех бордов: 0.5412 против одного -> 0.5765 против двух.
    # Тест несёт двойную нагрузку: наивная реализация, сэмплирующая оппонентов
    # независимо и не снимающая их карты из колоды, этот эффект не воспроизведёт.
    from harness.analysis.tools.equity import equity_vs_ranges

    ak = Range(weights={"AKo": 1.0, "AKs": 1.0})
    one = equity_vs_ranges(("Qs", "Qh"), [ak])
    two = equity_vs_ranges(("Qs", "Qh"), [ak, ak])
    assert two > one + 0.03  # эффект крупный, это не шум выборки
