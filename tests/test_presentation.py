"""Единый голос продукта (задача 17): каждый конструктор `presentation` — на слова
и числа, которые он обязан произвести, и на кнопки с правильным `callback_data`.

Два теста в этом файле проверяют не текст, а *отсутствие* определённого текста —
пометку зоны доверия и слово «ошибка». Оба построены на смеси строк ОБОИХ видов
(строгая + предполагающая зона; сообщение со скипом и без), чтобы шаблон,
безусловно печатающий (или безусловно не печатающий) нужную подстроку, не мог
пройти тест случайно — см. ловушку, которую описывает бриф задачи.

**Fix round 1.** Добавлены: `test_..._is_grammatically_correct` (пришпиливает
буквальный рендер строки решения — регресс формулировки «колл вместо шов»
пойман ревью, не тестом, который проверял только числа и пометку, но не саму
фразу вокруг них); `test_scan_summary_msg_total_loss_label_differs_from_items_sum`
(число в заголовке и сумма показанных строк — сознательно РАЗНЫЕ величины, и
предыдущая версия этого теста их не различала, потому что оба примера
совпадали численно); тесты на `hands_failed` и на не-минус-ноль `_fmt_bb`.
"""

from __future__ import annotations

from harness.analysis.scan import ScanItem, ScanSummary
from harness.contracts import (
    AnalysisResult,
    Assumption,
    PointVerdict,
    Range,
    SpotKind,
    Street,
    Zone,
)
from harness.presentation import (
    Btn,
    Msg,
    deep_dive_msg,
    escalation_msg,
    failed_msg,
    progress_text,
    quota_exceeded_msg,
    scan_summary_msg,
)

# --- progress_text -----------------------------------------------------------------


def test_progress_text_covers_all_four_stations():
    assert progress_text("parse") == "Читаю стол…"
    assert progress_text("validate") == "Проверяю руку…"
    assert progress_text("analyze") == "Считаю эквити…"
    assert progress_text("explain") == "Формулирую…"


# --- Msg/Btn — минимальная форма ----------------------------------------------------


def test_msg_defaults_to_no_buttons():
    assert Msg(text="привет").buttons == []


# --- scan_summary_msg ---------------------------------------------------------------


def _scan_item(*, hand_no: str, ev_diff_bb: float, zone: Zone, hero_class: str = "AA") -> ScanItem:
    return ScanItem(
        hand_no=hand_no,
        hand_index=1,
        hero_class=hero_class,
        spot=SpotKind.PUSHFOLD_UNOPENED,
        action_taken="fold",
        best_action="shove",
        ev_diff_bb=ev_diff_bb,
        zone=zone,
    )


def test_scan_summary_msg_lists_items_with_price_and_deep_dive_button():
    items = [
        _scan_item(hand_no="H1", ev_diff_bb=-2.3, zone=Zone.STRICT),
        _scan_item(hand_no="H2", ev_diff_bb=-1.1, zone=Zone.ASSUMING, hero_class="72o"),
    ]
    s = ScanSummary(hands_total=10, hands_with_decision=8, items=items, total_loss_bb=-3.4)

    msg = scan_summary_msg(s, quota_left=17, quota_total=50)

    assert "−2.3 bb" in msg.text
    assert "−1.1 bb" in msg.text
    assert f"{17}/{50}" in msg.text
    assert "за 24 ч" in msg.text
    assert len(msg.buttons) == 2
    assert msg.buttons[0] == [Btn(text="разобрать", callback_data="deep:H1")]
    assert msg.buttons[1] == [Btn(text="разобрать", callback_data="deep:H2")]


def test_scan_summary_msg_never_says_error_word_with_or_without_items():
    """CLAUDE.md: слово — «расхождение», никогда «ошибка» — ни в одной из двух веток."""
    with_items = ScanSummary(
        hands_total=5,
        hands_with_decision=5,
        items=[_scan_item(hand_no="H1", ev_diff_bb=-5.0, zone=Zone.ASSUMING)],
        total_loss_bb=-5.0,
    )
    without_items = ScanSummary(hands_total=5, hands_with_decision=5, items=[], total_loss_bb=0.0)

    # "ошиб" — общий корень всех падежей («ошибка», «ошибки», «ошибок» genitive
    # plural со сдвигом гласной и т.д.), а не только «ошибк»: более узкая
    # подстрока не поймала бы «ошибок» и дала бы тесту молча пройти мимо утечки
    # (найдено фальсификацией — см. отчёт задачи 17).
    assert "ошиб" not in scan_summary_msg(with_items, quota_left=1, quota_total=1).text
    assert "ошиб" not in scan_summary_msg(without_items, quota_left=1, quota_total=1).text
    assert "расхожд" in scan_summary_msg(with_items, quota_left=1, quota_total=1).text


def test_scan_summary_msg_marks_assuming_rows_and_not_strict_rows():
    """Гарантия честности зоны: пометка — у строки `assuming`, и ровно у неё.

    Обе строки в одной сводке различаются ценой (`−2.3 bb` / `−1.1 bb`), поэтому
    можно найти КОНКРЕТНУЮ строку каждого пункта и проверить пометку на ней, а
    не «где-то в тексте» — иначе шаблон, ставящий пометку на все строки без
    разбора (или ни на одну), прошёл бы проверку по ошибке.
    """
    items = [
        _scan_item(hand_no="H1", ev_diff_bb=-2.3, zone=Zone.STRICT),
        _scan_item(hand_no="H2", ev_diff_bb=-1.1, zone=Zone.ASSUMING, hero_class="72o"),
    ]
    s = ScanSummary(hands_total=10, hands_with_decision=8, items=items, total_loss_bb=-3.4)
    msg = scan_summary_msg(s, quota_left=17, quota_total=50)

    lines = msg.text.splitlines()
    strict_line = next(line for line in lines if "−2.3 bb" in line)
    assuming_line = next(line for line in lines if "−1.1 bb" in line)

    assert "по модели диапазонов" not in strict_line
    assert "по модели диапазонов" in assuming_line


def test_scan_summary_msg_no_items_has_no_buttons():
    s = ScanSummary(hands_total=3, hands_with_decision=1, items=[], total_loss_bb=0.0)
    msg = scan_summary_msg(s, quota_left=50, quota_total=50)
    assert msg.buttons == []


def test_scan_summary_msg_item_line_is_grammatically_correct():
    """Пришпиливает буквальный рендер — «вместо» требует родительного падежа
    («вместо шова», не «вместо шов»), первая версия строки была сломана
    именно на этом (fix round 1, Important 1); фраза переписана так, чтобы
    падеж вообще не был нужен («верно: шов»), и здесь это закреплено буквально,
    а не только проверкой чисел/пометки, которая ловушку не заметила.
    """
    items = [_scan_item(hand_no="H1", ev_diff_bb=-2.3, zone=Zone.STRICT)]
    s = ScanSummary(hands_total=1, hands_with_decision=1, items=items, total_loss_bb=-2.3)
    msg = scan_summary_msg(s, quota_left=1, quota_total=1)

    assert "№H1 · AA · пуш-фолд: фолд (верно: шов) — −2.3 bb" in msg.text
    assert "вместо шов" not in msg.text  # старая (сломанная) формулировка


def test_scan_summary_msg_total_loss_label_differs_from_items_sum():
    """`total_loss_bb` — по ВСЕМ судимым точкам файла, `items` — только дороже
    порога 0.1bb (докстринг `ScanSummary.total_loss_bb`) — на настоящем турнире
    первое обычно больше суммы вторых. Числа здесь НАРОЧНО не совпадают (единый
    видимый пункт −2.3bb против заголовочных −5.0bb), а подпись заголовка
    обязана явно называть его «по всем точкам разбора», а не пересказывать
    список ниже — иначе игрок видит два разных числа под одинаковой подписью
    и решает, что мы ошиблись в счёте (fix round 1, Important 2; докстринг
    `scan.py` — требование, которое бриф задачи 17 не унёс, ревью — унесло).
    """
    items = [_scan_item(hand_no="H1", ev_diff_bb=-2.3, zone=Zone.STRICT)]
    s = ScanSummary(hands_total=20, hands_with_decision=15, items=items, total_loss_bb=-5.0)
    msg = scan_summary_msg(s, quota_left=1, quota_total=1)

    assert "Суммарная потеря по всем точкам разбора: −5.0 bb." in msg.text
    assert "−2.3 bb" in msg.text  # цена одной показанной строки — другое число
    assert "Суммарная цена расхождений" not in msg.text  # старая (неточная) подпись


def test_scan_summary_msg_surfaces_hands_failed_when_nonzero():
    s = ScanSummary(
        hands_total=10, hands_with_decision=8, items=[], total_loss_bb=0.0, hands_failed=2
    )
    msg = scan_summary_msg(s, quota_left=1, quota_total=1)
    assert "Раздач не разобрано: 2" in msg.text


def test_scan_summary_msg_omits_hands_failed_line_when_zero():
    s = ScanSummary(
        hands_total=10, hands_with_decision=8, items=[], total_loss_bb=0.0, hands_failed=0
    )
    msg = scan_summary_msg(s, quota_left=1, quota_total=1)
    assert "не разобрано" not in msg.text


def test_scan_summary_msg_total_loss_near_zero_never_renders_negative_zero():
    """`_fmt_bb`: знак решается ПОСЛЕ округления, иначе `-0.03` → «−0.0 bb»,
    что читается как отдельная (мнимая) отрицательная величина (fix round 1)."""
    s = ScanSummary(hands_total=1, hands_with_decision=1, items=[], total_loss_bb=-0.03)
    msg = scan_summary_msg(s, quota_left=1, quota_total=1)
    assert "−0.0 bb" not in msg.text
    assert "0.0 bb" in msg.text


# --- deep_dive_msg -------------------------------------------------------------------


def _point(
    *, spot: SpotKind, ev_diff_bb: float, zone: Zone, assumption: Assumption | None = None
) -> PointVerdict:
    return PointVerdict(
        dp_index=0,
        street=Street.PREFLOP,
        spot=spot,
        zone=zone,
        action_taken="fold" if zone is Zone.STRICT else "call",
        best_action="shove",
        ev_diff_bb=ev_diff_bb,
        assumption=assumption,
    )


def _mixed_result(hand_no: str = "H42") -> AnalysisResult:
    strict_point = _point(spot=SpotKind.PUSHFOLD_UNOPENED, ev_diff_bb=-2.3, zone=Zone.STRICT)
    assuming_point = _point(
        spot=SpotKind.PUSHFOLD_FACING_SHOVE,
        ev_diff_bb=-1.1,
        zone=Zone.ASSUMING,
        assumption=Assumption(range=Range(weights={"AA": 1.0}), source="model:test"),
    )
    return AnalysisResult(
        hand_no=hand_no,
        points=[strict_point, assuming_point],
        ranked=[0, 1],
        total_ev_loss_bb=-3.4,
    )


def test_deep_dive_msg_has_status_line_with_zone_time_and_quota():
    res = _mixed_result()
    msg = deep_dive_msg(res, elapsed_s=12, zone=Zone.STRICT, quota_left=17, quota_total=50)
    assert "⏱ 12с" in msg.text
    assert "зона: строго" in msg.text
    assert "разборов 17/50 за 24 ч" in msg.text


def test_deep_dive_msg_status_line_shows_assuming_zone_word():
    res = _mixed_result()
    msg = deep_dive_msg(res, elapsed_s=7, zone=Zone.ASSUMING, quota_left=1, quota_total=50)
    assert "зона: предполагая" in msg.text


def test_deep_dive_msg_marks_assuming_points_and_not_strict_points():
    """Тот же honesty-инвариант, что и у скана, но на уровне точек решения руки."""
    res = _mixed_result()
    msg = deep_dive_msg(res, elapsed_s=12, zone=Zone.STRICT, quota_left=17, quota_total=50)

    lines = msg.text.splitlines()
    strict_line = next(line for line in lines if "−2.3 bb" in line)
    assuming_line = next(line for line in lines if "−1.1 bb" in line)

    assert "по модели диапазонов" not in strict_line
    assert "по модели диапазонов" in assuming_line


def test_deep_dive_msg_respects_ranked_order_not_points_order():
    """Порядок вывода — `res.ranked`, а не порядковый номер в `res.points`."""
    res = _mixed_result()
    res = res.model_copy(update={"ranked": [1, 0]})  # предполагающая точка первой
    msg = deep_dive_msg(res, elapsed_s=1, zone=Zone.STRICT, quota_left=1, quota_total=1)

    assuming_pos = msg.text.index("−1.1 bb")
    strict_pos = msg.text.index("−2.3 bb")
    assert assuming_pos < strict_pos


def test_deep_dive_msg_point_line_is_grammatically_correct():
    """Тот же пришпиленный рендер, что и у скана, — точка решения в разборе руки
    строится тем же f-строчным шаблоном и была сломана тем же образом.
    """
    res = _mixed_result()
    msg = deep_dive_msg(res, elapsed_s=1, zone=Zone.STRICT, quota_left=1, quota_total=1)

    assert "Префлоп · пуш-фолд: фолд (верно: шов) — −2.3 bb" in msg.text
    assert (
        "Префлоп · колл шова: колл (верно: шов) — −1.1 bb (по модели диапазонов)" in msg.text
    )
    assert "вместо шов" not in msg.text


def test_deep_dive_msg_buttons_are_ranges_detail_disagree_with_hand_no():
    res = _mixed_result(hand_no="H99")
    msg = deep_dive_msg(res, elapsed_s=1, zone=Zone.STRICT, quota_left=1, quota_total=1)
    assert len(msg.buttons) == 1
    row = msg.buttons[0]
    assert [b.text for b in row] == ["🎯 Диапазоны", "🔍 Подробнее", "✋ Не согласен"]
    assert [b.callback_data for b in row] == ["ranges:H99", "detail:H99", "disagree:H99"]


def test_deep_dive_msg_dev_line_appears_only_when_passed():
    res = _mixed_result()
    without = deep_dive_msg(res, elapsed_s=12, zone=Zone.STRICT, quota_left=1, quota_total=1)
    with_dev = deep_dive_msg(
        res,
        elapsed_s=12,
        zone=Zone.STRICT,
        quota_left=1,
        quota_total=1,
        dev_line="себестоимость: $0.0042, gpt-4o-mini",
    )
    assert "себестоимость" not in without.text
    assert "$0.0042" not in without.text
    assert "себестоимость: $0.0042, gpt-4o-mini" in with_dev.text


def test_deep_dive_msg_with_no_ranked_points_does_not_crash():
    res = AnalysisResult(hand_no="H0", points=[], ranked=[], total_ev_loss_bb=0.0)
    msg = deep_dive_msg(res, elapsed_s=3, zone=Zone.STRICT, quota_left=1, quota_total=1)
    assert "H0" in msg.text


# --- escalation_msg ------------------------------------------------------------------


def test_escalation_msg_has_option_buttons_plus_manual_entry():
    msg = escalation_msg(
        field="hero_stack_bb", question="Стек героя: 12.7bb?", options=["12.7", "12.1"]
    )
    assert msg.text == "Стек героя: 12.7bb?"
    assert len(msg.buttons) == 1
    row = msg.buttons[0]
    assert [b.text for b in row] == ["12.7", "12.1", "ввести вручную"]
    assert [b.callback_data for b in row] == [
        "escalate:hero_stack_bb:12.7",
        "escalate:hero_stack_bb:12.1",
        "escalate:hero_stack_bb:manual",
    ]


# --- failed_msg / quota_exceeded_msg --------------------------------------------------


def test_failed_msg_carries_the_public_reason_with_no_buttons():
    msg = failed_msg("стол расходится с движком по деньгам")
    assert "стол расходится с движком по деньгам" in msg.text
    assert msg.buttons == []


def test_quota_exceeded_msg_states_hours_and_window():
    msg = quota_exceeded_msg(hours_to_free=5)
    assert "5 ч" in msg.text
    assert "24 ч" in msg.text
    assert msg.buttons == []
