import json
import random

import pytest

from harness.analysis.tools.pushfold import (
    BRACKET_TIGHT,
    BRACKET_WIDE,
    CallerModel,
    call_shove_ev_bb,
    class_equity,
    default_call_prob,
    equity_vs_range_classes,
    fold_equity_ok,
    nash_hu,
    nash_hu_regret_bb,
    shove_ev_bb,
)
from harness.contracts import Range, all_classes, class_of


def test_shove_ev_headsup_computed():
    # Hero SB: стек 10bb, поставил 0.5 -> за спиной 9.5; BB: стек 10bb, поставил 1.0,
    # за спиной 9.0; в банке 1.5 — это ровно два поста, чужих денег в банке нет.
    # фолд-ветка (0.6): +1.5bb (весь банк, включая свои 0.5 — они в базлайне уже потеряны)
    # колл-ветка (0.4): вклады равны (10.0 и 10.0), непокрытого остатка нет,
    #   банк 0 + 10 + 10 = 20 -> 0.4*20 - (10 - 0.5) = -1.5bb
    # EV = 0.6*1.5 + 0.4*(-1.5) = +0.3bb
    caller = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0, posted_bb=1.0)
    ev = shove_ev_bb(
        "32o",
        hero_behind_bb=9.5,
        pot_dead_bb=1.5,
        callers=[caller],
        hero_posted_bb=0.5,
        equity_fn=lambda *a, **k: 0.40,
        call_prob_fn=lambda *a: 0.4,
    )
    assert abs(ev - 0.3) < 1e-9


def test_shove_ev_same_spot_without_posts_is_a_different_and_correct_number():
    # Тот же расклад, но посты не заданы (по умолчанию 0): тогда 1.5 в банке —
    # это чужие мёртвые деньги, вклад героя 9.5 против 9.0 коллера, и непокрытые
    # 0.5 герою возвращаются. Банк 1.5 + 9 + 9 = 19.5, вклад 0.4*19.5 - 9 = -1.2.
    # EV = 0.6*1.5 + 0.4*(-1.2) = +0.42. Это не расхождение с якорем выше, а
    # другая раздача: без постов модель обязана вернуть непокрытый остаток.
    caller = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0)
    ev = shove_ev_bb(
        "32o",
        hero_behind_bb=9.5,
        pot_dead_bb=1.5,
        callers=[caller],
        equity_fn=lambda *a, **k: 0.40,
        call_prob_fn=lambda *a: 0.4,
    )
    assert abs(ev - 0.42) < 1e-9


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


def test_shove_ev_short_caller_returns_hero_uncalled_remainder():
    # Регрессия на систематическое смещение, найденное ревью. Герой шовит 10bb,
    # единственный коллер может поставить только 4bb, чужих мёртвых денег 1.5.
    # Непокрытые 6bb шова герою ВОЗВРАЩАЮТСЯ: в контесте 1.5 + 4 + 4 = 9.5,
    # риск 4. Вклад = 0.5 * 9.5 - 4 = +0.75bb.
    # Прежняя формула (банк = pot_dead + hero_behind + min(behind_i, hero_behind))
    # давала -2.25bb — она считала непокрытый остаток проигранным.
    short = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=4.0)
    ev = shove_ev_bb(
        "KK",
        hero_behind_bb=10.0,
        pot_dead_bb=1.5,
        callers=[short],
        equity_fn=lambda *a, **k: 0.5,
        call_prob_fn=lambda *a: 1.0,
    )
    assert abs(ev - 0.75) < 1e-9


def test_shove_ev_short_caller_with_posts_accounts_them_in_totals():
    # Тот же по форме спот, но 1.5 в банке — это посты самих игроков, а не чужие
    # деньги: герой 0.5 + 10 = 10.5 вклада, коллер 1.0 + 3 = 4.0. Матчится 4.0,
    # чужих денег в банке нет: контест 0 + 4 + 4 = 8, риск сверх поста 4 - 0.5.
    # Вклад = 0.5 * 8 - 3.5 = +0.5bb — не +0.75: посты меняют, кто сколько внёс.
    short = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=3.0, posted_bb=1.0)
    ev = shove_ev_bb(
        "KK",
        hero_behind_bb=10.0,
        pot_dead_bb=1.5,
        callers=[short],
        hero_posted_bb=0.5,
        equity_fn=lambda *a, **k: 0.5,
        call_prob_fn=lambda *a: 1.0,
    )
    assert abs(ev - 0.5) < 1e-9


def test_shove_ev_at_risk_is_the_largest_caller_not_every_caller():
    # Позади короткий A (1.0 + 3.0 = 4.0) и покрывающий B (0 + 10.0), герой 0.5 + 9.5.
    # Сколько герой рискует, определяет САМЫЙ БОЛЬШОЙ вклад среди заколлировавших,
    # а каждый коллер вносит не больше своего вклада:
    #   ()     -> +1.5
    #   {A}    -> чужих 0, риск 4,  банк 4 + 4 = 8       -> 0.5*8 - 3.5  = +0.5
    #   {B}    -> чужих 1.0 (пост A остаётся в банке), риск 10,
    #             банк 1 + 10 + 10 = 21                   -> 0.5*21 - 9.5 = +1.0
    #   {A, B} -> чужих 0, риск 10, банк 10 + 4 + 10 = 24 -> 0.5*24 - 9.5 = +2.5
    short = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=3.0, posted_bb=1.0)
    covering = CallerModel(call_range=Range(weights={"KK": 1.0}), behind_bb=10.0)
    ev = shove_ev_bb(
        "QQ",
        hero_behind_bb=9.5,
        pot_dead_bb=1.5,
        callers=[short, covering],
        hero_posted_bb=0.5,
        equity_fn=lambda *a, **k: 0.5,
        call_prob_fn=lambda *a: 0.5,
    )
    assert abs(ev - 0.25 * (1.5 + 0.5 + 1.0 + 2.5)) < 1e-9


def test_shove_ev_rejects_posts_exceeding_the_pot():
    # Посты не могут быть больше банка: иначе "чужие деньги" в ветке уходят в минус,
    # и банк тихо занижается. Это ошибка вызывающей стороны, а не повод считать.
    caller = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0, posted_bb=1.0)
    with pytest.raises(ValueError, match="постов"):
        shove_ev_bb(
            "AA",
            hero_behind_bb=9.5,
            pot_dead_bb=1.0,
            callers=[caller],
            hero_posted_bb=0.5,
        )


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
    # без постов: банк 1.5 + 9.0 + 9.0 = 19.5, риск = min(9.5, 9.0) = 9.0
    p = 6 / 1225
    want = (1 - p) * 1.5 + p * (0.10 * 19.5 - 9.0)
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


def _wide_expected_at(depth: float) -> set[str]:
    """Независимый пересчёт состава широкой вилки: порядок по эквити ПРОТИВ ШОВА."""
    from harness.analysis.tools.equity import combos_of_class

    push, _ = nash_hu(depth)
    live = _live_combo_share()
    classes = all_classes()

    strength: dict[str, float] = {}
    for hero in classes:
        weighted = sum(
            live[(hero, h)] * push.weight(h) * class_equity(hero, h) for h in classes
        )
        mass = sum(live[(hero, h)] * push.weight(h) for h in classes)
        strength[hero] = weighted / mass
    ordered = sorted(classes, key=lambda c: (-strength[c], c))

    target = 0.40 * 1326
    cumulative, best_prefix, best_gap = 0, 0, target
    for position, cls in enumerate(ordered, start=1):
        cumulative += len(combos_of_class(cls))
        if abs(cumulative - target) < best_gap:
            best_gap, best_prefix = abs(cumulative - target), position
    return set(ordered[:best_prefix])


@pytest.mark.slow
def test_bracket_wide_is_ranked_by_equity_against_the_shoving_range():
    # Состав широкого конца вилки не выбирается руками И не ранжируется против
    # случайной руки: коллер отвечает на ШОВ, поэтому руки упорядочены по эквити
    # против равновесного диапазона шова на той же глубине. Тест воспроизводит
    # вывод независимо и требует совпадения состава.
    for depth in (5.0, 10.0):
        assert set(BRACKET_WIDE(depth).weights) == _wide_expected_at(depth), depth


@pytest.mark.slow
def test_small_pair_is_in_the_wide_bracket_at_every_depth():
    # Существенное последствие смены критерия: при ранжировании против СЛУЧАЙНОЙ
    # руки двойка вылетала из вилки (0.5036 при границе 0.5231), хотя любой
    # лузовый оппонент коллирует шов с 22. Против диапазона шова она проходит
    # на всех глубинах — вилка снова ограничивает поведение оппонента сверху.
    for depth in (5.0, 10.0, 15.0, 20.0):
        assert "22" in BRACKET_WIDE(depth).weights, depth


@pytest.mark.slow
def test_small_pair_vs_weak_ace_ordering_flips_with_the_width_of_the_shove():
    # Порядок «22 против A2o» не абсолютный, он зависит от ШИРИНЫ диапазона шова,
    # и это проверяется на обоих концах, чтобы фиксировать механизм, а не якорь.
    # Против узкого шова (25bb, 36% комбо) впереди пара: узкий шов — это в основном
    # старшие карты и пары, слабый туз там доминирован. Против широкого (10bb,
    # 58% комбо) впереди туз: в широком диапазоне много несвязанного мусора, против
    # которого туз-хай выигрывает, а двойка остаётся позади любой старшей пары.
    narrow, _ = nash_hu(25.0)
    wide, _ = nash_hu(10.0)
    assert narrow.fraction_of_hands() < 0.40 < wide.fraction_of_hands()

    assert equity_vs_range_classes("22", narrow) > equity_vs_range_classes("A2o", narrow)
    assert equity_vs_range_classes("22", wide) < equity_vs_range_classes("A2o", wide)


@pytest.mark.slow
def test_bracket_wide_leaves_suited_connectors_out():
    # Не регресс, а свойство: колл шова — это вскрытие, а разыгрываемость, ради
    # которой одномастные коннекторы держат, там не существует. Они вне вилки и
    # по старому критерию, и по новому.
    wide = BRACKET_WIDE(10.0).weights
    for cls in ("65s", "76s", "87s"):
        assert cls not in wide, cls


@pytest.mark.slow
def test_bracket_wide_moves_with_depth():
    # Диапазон шова на 5bb заметно шире, чем на 15bb, поэтому и порядок сил
    # против него другой: состав вилки обязан зависеть от глубины, иначе аргумент
    # "ранжируем против шова" не работает.
    assert set(BRACKET_WIDE(5.0).weights) != set(BRACKET_WIDE(15.0).weights)


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


# --- Таблица эквити 169x169 и HU-равновесие -----------------------------------


def _live_combo_share() -> dict[tuple[str, str], float]:
    """P(у оппонента класс c | у игрока класс h) — независимый перебор комбо.

    Пересчитывает блокеры в лоб, не пользуясь ничем из pushfold: если реализация
    ошибётся в снятии карт, тесты, опирающиеся на этот словарь, разойдутся с ней.
    """
    from harness.analysis.tools.equity import combos_of_class

    classes = all_classes()
    combos = {cls: combos_of_class(cls) for cls in classes}
    return {
        (h, c): sum(
            1 for a in combos[h] for b in combos[c] if a[0] not in b and a[1] not in b
        )
        / len(combos[h])
        / 1225
        for h in classes
        for c in classes
    }



def test_class_equity_is_symmetric_with_half_on_diagonal():
    # Эквити — доля банка в хедз-апе, поэтому eq(a,b) + eq(b,a) = 1 тождественно,
    # а класс против самого себя даёт ровно 0.5 по симметрии мастей.
    classes = all_classes()
    assert len(classes) == 169
    for a in classes:
        assert class_equity(a, a) == 0.5
    for a in classes[::17]:
        for b in classes:
            assert abs(class_equity(a, b) + class_equity(b, a) - 1.0) < 1e-9


def test_class_equity_matches_known_anchors():
    # Те же якоря, что уже проверены на равном движке эквити (tests/test_equity.py):
    # они ловят перепутанные оси таблицы и сдвиг индексов.
    assert abs(class_equity("AA", "KK") - 0.815) < 0.02
    assert abs(class_equity("AKs", "QQ") - 0.46) < 0.015
    assert class_equity("AA", "72o") > 0.85
    assert class_equity("72o", "AA") < 0.15


def test_class_equity_aces_beat_everything():
    for other in all_classes():
        if other == "AA":
            continue
        assert class_equity("AA", other) > 0.65, other


def test_class_equity_rejects_unknown_class():
    with pytest.raises(ValueError, match="класс"):
        class_equity("AA", "AAs")


@pytest.mark.slow
def test_class_equity_matches_independent_recomputation():
    # Обязательная сверка закоммиченных данных: таблица считалась один раз скриптом,
    # и больше её никто не пересчитает. Здесь несколько клеток пересчитываются
    # НЕЗАВИСИМЫМ путём — через equity_vs_range, другим сидом и другим числом
    # итераций. Совпадение в пределах допуска означает, что данные не подогнаны
    # и оси не перепутаны.
    from harness.analysis.tools.equity import equity_vs_range
    from harness.analysis.tools.pushfold import representative_combo

    rng = random.Random(20260830)
    classes = all_classes()
    cells = [(rng.choice(classes), rng.choice(classes)) for _ in range(6)]
    for hero_cls, villain_cls in cells:
        hero = representative_combo(hero_cls)
        villain_range = Range(weights={villain_cls: 1.0})
        want = equity_vs_range(hero, villain_range, iterations=30_000, seed=987_654_321)
        got = class_equity(hero_cls, villain_cls)
        assert abs(got - want) < 0.02, (hero_cls, villain_cls, got, want)


@pytest.mark.slow
def test_nash_hu_anchors_10bb(tmp_path):
    push, call = nash_hu(10.0, cache_dir=tmp_path)
    assert push.weight("AA") == 1.0 and call.weight("AA") == 1.0  # AA всегда
    assert push.weight("22") > 0.9  # мелкие пары пушатся на 10bb
    assert push.weight("32o") < 0.1  # мусор — фолд на 10bb
    assert call.weight("32o") < 0.05  # и тем более не колл


@pytest.mark.slow
def test_nash_monotone_by_depth(tmp_path):
    p5, c5 = nash_hu(5.0, cache_dir=tmp_path)
    p10, c10 = nash_hu(10.0, cache_dir=tmp_path)
    p15, c15 = nash_hu(15.0, cache_dir=tmp_path)
    assert p5.fraction_of_hands() >= p10.fraction_of_hands()  # мельче — шире
    assert p10.fraction_of_hands() >= p15.fraction_of_hands()
    assert c5.fraction_of_hands() >= c10.fraction_of_hands() >= c15.fraction_of_hands()


@pytest.mark.slow
def test_nash_push_is_wider_than_call():
    # SB рискует стеком, чтобы забрать 1bb блайндов, поэтому пушит заметно шире,
    # чем BB коллирует: у BB нет фолд-эквити, только вскрытие.
    push, call = nash_hu(10.0)
    assert push.fraction_of_hands() > call.fraction_of_hands() + 0.1


@pytest.mark.slow
def test_nash_at_2bb_matches_closed_form_threshold():
    # На 2bb равновесие раскрывается в лоб, и это независимая от fictitious play
    # проверка ВСЕХ 169 классов сразу, а не якорь на пару рук.
    # BB: доплатить 1 в банк 4 -> нужно 25% эквити; худшая рука против ~90%
    # диапазона держит около 32%, поэтому BB коллирует любые две карты.
    # SB против 100% колла: шов даёт 2*2*eq - 2, фолд -0.5 => пуш ровно при
    # eq > 0.375 против случайной руки.
    push, call = nash_hu(2.0)
    assert call.fraction_of_hands() == 1.0

    classes = all_classes()
    live = _live_combo_share()
    borderline = 0
    for hero in classes:
        eq_vs_random = sum(live[(hero, c)] * class_equity(hero, c) for c in classes)
        if abs(eq_vs_random - 0.375) < 0.005:
            borderline += 1  # в полосе шума таблицы (SE ~0.15%) знак не определён
            continue
        expected = 1.0 if eq_vs_random > 0.375 else 0.0
        assert abs(push.weight(hero) - expected) < 0.05, (hero, eq_vs_random, push.weight(hero))
    assert borderline <= 5  # полоса неопределённости должна быть узкой


def test_nash_cached_deterministic(tmp_path):
    a, _ = nash_hu(8.0, cache_dir=tmp_path)
    b, _ = nash_hu(8.0, cache_dir=tmp_path)
    assert a == b


def test_nash_cache_is_speedup_not_source_of_truth(tmp_path):
    # Кэш обязан совпадать с расчётом с нуля: тест проходит и на пустом кэше.
    fresh_a = nash_hu(6.0, cache_dir=tmp_path / "a")
    assert (tmp_path / "a" / "nash_hu_6.00.json").exists()
    fresh_b = nash_hu(6.0, cache_dir=tmp_path / "b")  # другой пустой каталог
    cached = nash_hu(6.0, cache_dir=tmp_path / "a")  # тот же, но уже с файлом
    assert fresh_a == fresh_b == cached


@pytest.mark.slow
def test_nash_guarantees_per_hand_regret_not_just_aggregate():
    # Агрегатная эксплуатируемость взвешена априорной вероятностью класса, поэтому
    # редкий класс может быть плох сам по себе и почти не сдвинуть сумму. Продукт
    # же читает этот модуль ПОКЛАССНО, поэтому гарантия нужна на класс, а не на
    # среднее. Тест пересобирает игру независимо и берёт МАКСИМУМ по классам.
    from harness.analysis.tools.equity import combos_of_class

    live = _live_combo_share()
    classes = all_classes()
    for eff in (5.0, 10.0, 15.0):
        push, call = nash_hu(eff)
        shove_ev = {
            h: sum(
                live[(h, c)]
                * (
                    call.weight(c) * (2.0 * class_equity(h, c) - 1.0) * eff
                    + (1.0 - call.weight(c)) * 1.0
                )
                for c in classes
            )
            for h in classes
        }
        worst_sb = max(
            max(shove_ev[h], -0.5) - (push.weight(h) * shove_ev[h] + (1 - push.weight(h)) * -0.5)
            for h in classes
        )
        diff = {
            b: sum(
                live[(b, h)] * push.weight(h) * (2.0 * class_equity(b, h) * eff - eff + 1.0)
                for h in classes
            )
            for b in classes
        }
        worst_bb = max(max(diff[b], 0.0) - call.weight(b) * diff[b] for b in classes)
        assert max(worst_sb, worst_bb) <= 0.01, (eff, worst_sb, worst_bb)
        assert combos_of_class("AA")  # диапазоны непусты, тест не вырожден


@pytest.mark.slow
def test_nash_fractional_weight_implies_near_indifference():
    # Прямая проверка свойства, ради которого нужна пер-хендовая граница: если
    # класс вернулся с ДРОБНЫМ весом, он обязан быть почти безразличен, а не быть
    # остатком усреднения на явно плюсовой руке. Арифметика: регрет = (вес на
    # худшем действии) * отрыв, поэтому регрет <= 0.01 и вес >= 0.1 дают отрыв
    # <= 0.1bb. Задача 12 не должна принимать "пушит 40% времени" за безразличие,
    # когда на деле это чистый пуш.
    live = _live_combo_share()
    classes = all_classes()
    eff = 10.0
    push, call = nash_hu(eff)

    for h in classes:
        weight = push.weight(h)
        if not 0.1 <= weight <= 0.9:
            continue
        shove_ev = sum(
            live[(h, c)]
            * (
                call.weight(c) * (2.0 * class_equity(h, c) - 1.0) * eff
                + (1.0 - call.weight(c)) * 1.0
            )
            for c in classes
        )
        assert abs(shove_ev - (-0.5)) <= 0.1, (h, weight, shove_ev)


@pytest.mark.slow
def test_nash_exposes_achieved_per_hand_regret():
    # Задача 12 должна уметь отказаться от «строго», поэтому достигнутая граница —
    # часть выдачи, а не внутренняя деталь.
    bound = nash_hu_regret_bb(10.0)
    assert 0.0 < bound <= 0.01


def test_nash_raises_instead_of_returning_unconverged(monkeypatch, tmp_path):
    # Молчаливый возврат последнего среднего при исчерпании лимита — ровно тот
    # отказ («вернуть правдоподобное вместо падения»), который в этом проекте
    # ловится снова и снова.
    from harness.analysis.tools import pushfold

    monkeypatch.setattr(pushfold, "_FP_MAX_ITERATIONS", 5)
    with pytest.raises(RuntimeError, match="не сошёлся"):
        nash_hu(10.0, cache_dir=tmp_path)


def test_nash_cache_recomputes_on_foreign_fingerprint(tmp_path):
    # Кэш переживает перегенерацию таблицы эквити и смену порогов, поэтому без
    # отпечатка он тихо отдаёт равновесие, посчитанное на других входных данных.
    push, _ = nash_hu(9.0, cache_dir=tmp_path)
    path = tmp_path / "nash_hu_9.00.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fingerprint"] = "чужой-отпечаток"
    payload["push"] = {"AA": 1.0}  # заведомо неверное равновесие
    path.write_text(json.dumps(payload), encoding="utf-8")

    again, _ = nash_hu(9.0, cache_dir=tmp_path)
    assert again == push
    assert json.loads(path.read_text(encoding="utf-8"))["fingerprint"] != "чужой-отпечаток"


def test_nash_cache_recomputes_when_stored_depth_disagrees(tmp_path):
    # Имя файла округляет глубину до сотых, поэтому одного имени мало: сверяется
    # ещё и записанный внутрь eff_bb.
    push, _ = nash_hu(9.0, cache_dir=tmp_path)
    path = tmp_path / "nash_hu_9.00.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["eff_bb"] = 20.0
    payload["push"] = {"AA": 1.0}
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert nash_hu(9.0, cache_dir=tmp_path)[0] == push


def test_nash_rejects_impossible_depth(tmp_path):
    with pytest.raises(ValueError, match="эффективный стек"):
        nash_hu(0.5, cache_dir=tmp_path)


@pytest.mark.slow
def test_nash_is_mutual_best_response():
    # Оракул равновесия по определению, а не по якорям: пересобираем игру с нуля
    # (включая условное распределение руки оппонента по блокерам — прямым перебором
    # комбо) и проверяем, что ни SB, ни BB не выигрывают от отклонения. Такой тест
    # ловит и перепутанные знаки в EV, и ошибку в блокерной матрице, и недосходимость.
    from harness.analysis.tools.equity import combos_of_class

    eff = 10.0
    push, call = nash_hu(eff)
    classes = all_classes()
    live = _live_combo_share()
    prior = {cls: len(combos_of_class(cls)) / 1326 for cls in classes}

    # SB: EV шова против средней стратегии колла BB; альтернатива — фолд за -0.5.
    shove_ev = {
        h: sum(
            live[(h, c)]
            * (
                call.weight(c) * (2.0 * class_equity(h, c) - 1.0) * eff
                + (1.0 - call.weight(c)) * 1.0
            )
            for c in classes
        )
        for h in classes
    }
    sb_gain = sum(
        prior[h]
        * (
            max(shove_ev[h], -0.5)
            - (push.weight(h) * shove_ev[h] + (1.0 - push.weight(h)) * -0.5)
        )
        for h in classes
    )

    # BB: разница "колл минус фолд" против средней стратегии шова SB.
    call_minus_fold = {
        b: sum(
            live[(b, h)] * push.weight(h) * (2.0 * class_equity(b, h) * eff - eff + 1.0)
            for h in classes
        )
        for b in classes
    }
    bb_gain = sum(
        prior[b] * (max(call_minus_fold[b], 0.0) - call.weight(b) * call_minus_fold[b])
        for b in classes
    )

    # Порог равен критерию остановки fictitious play (_FP_EXPLOITABILITY_BB = 1e-3
    # на двоих) — тест проверяет, что заявленная сходимость действительно достигнута,
    # и меряет это независимо пересобранной игрой.
    assert 0.0 <= sb_gain + bb_gain < 1e-3, (sb_gain, bb_gain)
