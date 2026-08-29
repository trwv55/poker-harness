import pytest

from harness.analysis.tools.pushfold import (
    BRACKET_TIGHT,
    BRACKET_WIDE,
    CallerModel,
    call_shove_ev_bb,
    default_call_prob,
    fold_equity_ok,
    shove_ev_bb,
)
from harness.contracts import Range, class_of


def test_shove_ev_headsup_computed():
    # Hero SB: стек 10bb, поставил 0.5 -> за спиной 9.5; BB за спиной 9.0; в банке 1.5
    # фолд-ветка (0.6): +1.5bb (весь банк, включая свои 0.5 — они в базлайне уже потеряны)
    # колл-ветка (0.4): банк 1.5+9.5+9.0=20 -> 0.4*20 - 9.5 = -1.5bb
    # EV = 0.6*1.5 + 0.4*(-1.5) = +0.3bb
    caller = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0)
    ev = shove_ev_bb(
        "32o",
        hero_behind_bb=9.5,
        pot_dead_bb=1.5,
        callers=[caller],
        equity_fn=lambda *a, **k: 0.40,
        call_prob_fn=lambda *a: 0.4,
    )
    assert abs(ev - 0.3) < 1e-9


def test_shove_ev_multiway_enumerates_subsets():
    # два игрока позади, каждый коллит с p=0.5; эквити героя при любом колле = 0
    # оба фолдят (0.25): +2.0bb; любая колл-ветка (0.75): -9.0bb (весь стек за спиной)
    c = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0)
    ev = shove_ev_bb(
        "32o",
        hero_behind_bb=9.0,
        pot_dead_bb=2.0,
        callers=[c, c],
        equity_fn=lambda *a, **k: 0.0,
        call_prob_fn=lambda *a: 0.5,
    )
    assert abs(ev - (0.25 * 2.0 + 0.75 * -9.0)) < 1e-9  # перебор 4 веток, не попарно


def test_shove_ev_everyone_folds_wins_whole_dead_pot():
    # p_call = 0 -> одна ветка "все сфолдили", вклад ровно pot_dead_bb
    c = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=50.0)
    ev = shove_ev_bb(
        "72o",
        hero_behind_bb=12.0,
        pot_dead_bb=2.25,
        callers=[c, c, c],
        equity_fn=lambda *a, **k: 0.0,
        call_prob_fn=lambda *a: 0.0,
    )
    assert abs(ev - 2.25) < 1e-9


def test_shove_ev_caller_shorter_than_hero_contributes_only_his_stack():
    # Коллер за спиной держит 4bb против 9bb героя: в контест входит min(4, 9) = 4,
    # остаток его стека в банк не идёт. Банк = 1.5 + 9 + 4 = 14.5.
    # p_call = 1 -> ветка одна: 0.5 * 14.5 - 9 = -1.75
    short = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=4.0)
    ev = shove_ev_bb(
        "KK",
        hero_behind_bb=9.0,
        pot_dead_bb=1.5,
        callers=[short],
        equity_fn=lambda *a, **k: 0.5,
        call_prob_fn=lambda *a: 1.0,
    )
    assert abs(ev - (0.5 * 14.5 - 9.0)) < 1e-9


def test_shove_ev_passes_only_calling_ranges_to_equity_fn():
    # equity_fn должна видеть ровно диапазоны тех, кто заколлировал в данной ветке,
    # и конкретное комбо героя (не класс) — блокеры считаются по картам.
    a = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=10.0)
    k = CallerModel(call_range=Range(weights={"KK": 1.0}), behind_bb=10.0)
    seen: list[tuple[tuple[str, str], tuple[str, ...]]] = []

    def spy(hero_combo, ranges, *args, **kwargs):
        classes = tuple(sorted(next(iter(r.weights)) for r in ranges))
        seen.append((hero_combo, classes))
        return 0.5

    shove_ev_bb(
        "QQ",
        hero_behind_bb=10.0,
        pot_dead_bb=1.5,
        callers=[a, k],
        equity_fn=spy,
        call_prob_fn=lambda *args: 0.5,
    )

    assert {classes for _, classes in seen} == {("AA",), ("KK",), ("AA", "KK")}
    for hero_combo, _ in seen:
        assert class_of(*hero_combo) == "QQ"  # раскрытие класса в комбо, а не строка "QQ"


def test_shove_ev_prunes_negligible_branches():
    # Три коллера с p=0.001: ветка "коллят все трое" имеет вероятность 1e-9 < 1e-4
    # и не должна ни считаться, ни звать equity_fn (иначе 2^n дорогих вызовов MC).
    c = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=10.0)
    subsets: list[int] = []

    def spy(hero_combo, ranges, *args, **kwargs):
        subsets.append(len(ranges))
        return 0.5

    shove_ev_bb(
        "72o",
        hero_behind_bb=10.0,
        pot_dead_bb=1.5,
        callers=[c, c, c],
        equity_fn=spy,
        call_prob_fn=lambda *args: 0.001,
    )
    assert max(subsets) == 1  # ветки на 2 и 3 коллеров (1e-6 и 1e-9) отброшены


def test_shove_ev_rejects_too_many_callers():
    c = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=10.0)
    with pytest.raises(ValueError, match="7"):
        shove_ev_bb("AA", hero_behind_bb=10.0, pot_dead_bb=1.5, callers=[c] * 8)


def test_shove_ev_rejects_nonpositive_stack():
    c = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=10.0)
    with pytest.raises(ValueError, match="за спиной"):
        shove_ev_bb("AA", hero_behind_bb=0.0, pot_dead_bb=1.5, callers=[c])


def test_default_call_prob_is_share_of_combos_with_hero_blockers():
    # Диапазон колла = только AA. Против героя без тузов живы все 6 комбо AA: 6/1225.
    # У героя с AA два туза сняты — остаётся ровно 1 комбо AA: 1/1225.
    caller = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=10.0)
    assert abs(default_call_prob(caller, "72o") - 6 / 1225) < 1e-12
    assert abs(default_call_prob(caller, "AA") - 1 / 1225) < 1e-12
    assert abs(default_call_prob(caller, "AKo") - 3 / 1225) < 1e-12


def test_default_call_prob_respects_partial_weights():
    caller = CallerModel(call_range=Range(weights={"AA": 0.5, "KK": 1.0}), behind_bb=10.0)
    assert abs(default_call_prob(caller, "72o") - (0.5 * 6 + 6) / 1225) < 1e-12


def test_shove_ev_default_call_prob_is_used_when_not_injected():
    # Без call_prob_fn вероятность колла берётся из диапазона: AA -> 6/1225.
    caller = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0)
    ev = shove_ev_bb(
        "72o",
        hero_behind_bb=9.5,
        pot_dead_bb=1.5,
        callers=[caller],
        equity_fn=lambda *a, **k: 0.10,
    )
    p = 6 / 1225
    want = (1 - p) * 1.5 + p * (0.10 * 20.0 - 9.5)
    assert abs(ev - want) < 1e-12


def test_call_shove_ev_zero_at_pot_odds_breakeven():
    from harness.analysis.tools.pot_odds import required_equity

    pot, to_call = 20, 9
    need = required_equity(to_call, pot)
    ev = call_shove_ev_bb(
        "QQ",
        hero_bb=15.0,
        shover_range=Range(weights={"AA": 1.0}),
        pot_bb=float(pot),
        to_call_bb=float(to_call),
        equity_fn=lambda *a, **k: need,
    )
    assert abs(ev) < 1e-9

    better = call_shove_ev_bb(
        "QQ",
        hero_bb=15.0,
        shover_range=Range(weights={"AA": 1.0}),
        pot_bb=float(pot),
        to_call_bb=float(to_call),
        equity_fn=lambda *a, **k: need + 0.1,
    )
    assert better > 0


def test_call_shove_ev_returns_uncalled_excess_when_hero_is_shorter():
    # Шов 12bb в банк 13.5, у героя за спиной только 5: он коллит на 5, лишние 7
    # шоверу возвращаются и в контест не входят. Банк = 13.5 - 7 + 5 = 11.5.
    ev = call_shove_ev_bb(
        "QQ",
        hero_bb=5.0,
        shover_range=Range(weights={"AA": 1.0}),
        pot_bb=13.5,
        to_call_bb=12.0,
        equity_fn=lambda *a, **k: 0.5,
    )
    assert abs(ev - (0.5 * 11.5 - 5.0)) < 1e-9


def test_fold_equity_ok_true_for_real_ranges():
    tight = CallerModel(call_range=BRACKET_TIGHT(10.0), behind_bb=10.0)
    wide = CallerModel(call_range=BRACKET_WIDE(10.0), behind_bb=10.0)
    assert fold_equity_ok([tight])
    assert fold_equity_ok([tight, wide])
    assert fold_equity_ok([])  # сзади никого — коллировать некому


def test_fold_equity_ok_false_when_someone_calls_any_two():
    from harness.contracts import all_classes

    any_two = CallerModel(
        call_range=Range(weights=dict.fromkeys(all_classes(), 1.0)), behind_bb=10.0
    )
    tight = CallerModel(call_range=BRACKET_TIGHT(10.0), behind_bb=10.0)
    assert not fold_equity_ok([any_two])
    assert not fold_equity_ok([tight, any_two])  # хватает одного, кто не фолдит никогда


def test_bracket_tight_is_premium_only():
    tight = BRACKET_TIGHT(10.0)
    assert set(tight.weights) == {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
    assert all(w == 1.0 for w in tight.weights.values())
    assert abs(tight.fraction_of_hands() - (4 * 6 + 4 + 12) / 1326) < 1e-12


def test_bracket_wide_is_about_top_40_percent():
    wide = BRACKET_WIDE(10.0)
    assert abs(wide.fraction_of_hands() - 0.40) < 0.01
    assert all(w == 1.0 for w in wide.weights.values())


def test_bracket_wide_contains_tight_at_every_depth():
    for depth in (3.0, 8.0, 12.0, 20.0):
        tight, wide = BRACKET_TIGHT(depth), BRACKET_WIDE(depth)
        assert set(tight.weights) <= set(wide.weights)
        assert wide.fraction_of_hands() > tight.fraction_of_hands()


def test_shove_ev_monotone_in_equity_and_in_call_prob():
    c = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=10.0)
    kwargs = {"hero_behind_bb": 10.0, "pot_dead_bb": 1.5, "callers": [c]}
    half = {"call_prob_fn": lambda *a: 0.5}
    low = shove_ev_bb("72o", equity_fn=lambda *a, **k: 0.2, **half, **kwargs)
    high = shove_ev_bb("72o", equity_fn=lambda *a, **k: 0.6, **half, **kwargs)
    assert high > low
    # при эквити ниже безубытка чаще коллируют — хуже герою
    rare = shove_ev_bb(
        "72o", equity_fn=lambda *a, **k: 0.2, call_prob_fn=lambda *a: 0.1, **kwargs
    )
    assert rare > low
