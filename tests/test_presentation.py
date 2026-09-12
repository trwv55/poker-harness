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

**Fix round 2.** Round 1 заменило «вместо» (грамматика) на «верно» — падеж
решён, честность нет: «верно» заявляет, что сыгранное было неправильным, а в
зоне `assuming` ядро знает только более высокую EV при угаданном диапазоне.
Слово заменено на «лучше» (истинно в обеих зонах). Guard на «ошиб» (round 1)
ловит одно написание идеи «не называть решение неправильным»; добавлен
`test_scan_summary_msg_never_asserts_the_taken_action_was_wrong` — guard на
«верно» (ловит и «неверно» как подстроку) — второе написание той же идеи.
"""

from __future__ import annotations

import re

from harness.analysis import analyze_hand
from harness.contracts import (
    RIVER_CALL_DETAIL,
    TURN_FLOP_CALL_DETAIL,
    AnalysisResult,
    Assumption,
    EvInterval,
    PointVerdict,
    Range,
    ScanItem,
    ScanSummary,
    SpotKind,
    Street,
    Zone,
)
from harness.contracts.enriched import hero_stack_delta_bb
from harness.explanation import HandReplay, ReplaySpan
from harness.presentation import (
    Btn,
    Msg,
    bot_failure_msg,
    escalation_msg,
    failed_msg,
    hand_analysis_msgs,
    hh_accepted_msg,
    hh_scan_in_progress_msg,
    new_session_msg,
    progress_text,
    quota_exceeded_msg,
    scan_summary_msg,
    start_msg,
    unsupported_document_msg,
)
from tests.test_hand_replay import _postflop_hand, _two_side_pots_hand

# --- progress_text -----------------------------------------------------------------


def test_progress_text_covers_every_station():
    """Каждая станция конвейера (`worker.pipeline._Station`) имеет свою строку.

    Станций пять, строк четыре: `read` (скрин) и `parse` (файл раздач) говорят
    игроку одно и то же — он в обоих случаях ждёт чтения стола. Станции без
    строки не бывает: `progress_text` брал бы по ключу, которого нет.
    """
    from harness.presentation.messages import _STATION_TEXT

    assert progress_text("ask") == "Считаю по вашим раздачам…"
    assert progress_text("read") == "Читаю стол…"
    assert progress_text("parse") == "Читаю стол…"
    assert progress_text("validate") == "Проверяю руку…"
    assert progress_text("analyze") == "Считаю эквити…"
    assert "explain" not in _STATION_TEXT, "станции формулировки в конвейере больше нет"


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


def test_scan_summary_msg_never_asserts_the_taken_action_was_wrong():
    """Fix round 2: тот же манёвр, что запрещает «расхождение, не ошибка», под
    другим словом — «верно»/«неверно» заявляют, что сыгранное было неправильным,
    но в зоне `assuming` (58 из 77 судимых точек продукта на измеренных данных)
    ядро знает только, что альтернатива выигрывала EV ПРИ УГАДАННОМ диапазоне,
    не то, что сыгранное было ошибкой — маркер `_ASSUMING_MARKER` двумя строками
    выше специально это оговаривает, и «верно» рядом с ним его отменяет.

    Guard на «ошиб» (round 1) ловит одно написание идеи; этот — другое написание
    той же идеи («верно»/«неверно» — второе ловится как подстрока первого).
    Собран на смеси `assuming`+пустой сводки, как и парный тест на «ошиб» —
    чтобы шаблон, безусловно содержащий (или не содержащий) слово, не прошёл
    случайно.
    """
    with_items = ScanSummary(
        hands_total=5,
        hands_with_decision=5,
        items=[_scan_item(hand_no="H1", ev_diff_bb=-5.0, zone=Zone.ASSUMING)],
        total_loss_bb=-5.0,
    )
    without_items = ScanSummary(hands_total=5, hands_with_decision=5, items=[], total_loss_bb=0.0)

    assert "верно" not in scan_summary_msg(with_items, quota_left=1, quota_total=1).text
    assert "верно" not in scan_summary_msg(without_items, quota_left=1, quota_total=1).text
    assert "лучше" in scan_summary_msg(with_items, quota_left=1, quota_total=1).text


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
    падеж вообще не был нужен («лучше: шов»), и здесь это закреплено буквально,
    а не только проверкой чисел/пометки, которая ловушку не заметила.

    Слово — «лучше», не «верно» (fix round 2): «верно» заявляло бы, что
    сыгранное было неправильным, а ядро в зоне `assuming` знает только более
    высокую EV при угаданном диапазоне, не факт правильности альтернативы.
    """
    items = [_scan_item(hand_no="H1", ev_diff_bb=-2.3, zone=Zone.STRICT)]
    s = ScanSummary(hands_total=1, hands_with_decision=1, items=items, total_loss_bb=-2.3)
    msg = scan_summary_msg(s, quota_left=1, quota_total=1)

    assert "№H1 · AA · пуш-фолд: фолд (лучше: шов) — −2.3 bb" in msg.text
    assert "вместо шов" not in msg.text  # старая (сломанная) формулировка round 1
    assert "верно" not in msg.text  # старая (нечестная в assuming) формулировка round 2


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


def test_scan_summary_msg_caps_the_list_and_says_how_many_were_found():
    """Round 5, Item J: список пунктов ничем не ограничивался, а Телеграм режет
    `sendMessage` на 4096 символах — при ~85 символах на строку сообщение
    ломалось бы примерно с 48-го пункта. И ломалось бы дорого:
    `TelegramSender.send` делает `raise_for_status()`, задача уходит в ретрай, и
    игрок платит тремя полными пересканами файла, прежде чем услышит хоть что-то.

    Обрезка обязана быть ГРОМКОЙ: молча показать 20 из 60 — та же деградация без
    огласки, против которой уже стоит `hands_failed`.
    """
    items = [
        _scan_item(hand_no=f"H{i}", ev_diff_bb=-10.0 + i * 0.01, zone=Zone.STRICT)
        for i in range(60)
    ]
    s = ScanSummary(hands_total=200, hands_with_decision=180, items=items, total_loss_bb=-120.0)

    msg = scan_summary_msg(s, quota_left=1, quota_total=5)

    item_lines = [line for line in msg.text.splitlines() if line.startswith("№")]
    assert len(item_lines) == 20
    assert len(msg.buttons) == 20  # кнопка ровно под каждой показанной строкой
    assert "20" in msg.text and "60" in msg.text, "сколько показано и сколько найдено"
    assert len(msg.text) < 4096  # телеграмный предел одного сообщения
    # Показаны самые дорогие: `items` приходят отсортированными по цене.
    assert item_lines[0].startswith("№H0")
    assert "№H59" not in msg.text


def test_scan_summary_msg_does_not_mention_a_cap_it_did_not_apply():
    """Обратная сторона: пока пунктов меньше потолка, заголовок обязан остаться
    прежним. Измеренные фикстуры дают девять — типичный случай не должен
    обрастать оговоркой про обрезку, которой не было.
    """
    items = [_scan_item(hand_no=f"H{i}", ev_diff_bb=-1.0, zone=Zone.STRICT) for i in range(9)]
    s = ScanSummary(hands_total=146, hands_with_decision=100, items=items, total_loss_bb=-9.0)

    msg = scan_summary_msg(s, quota_left=1, quota_total=5)

    assert "Топ расхождений:" in msg.text
    assert "показаны" not in msg.text.lower()
    assert len([line for line in msg.text.splitlines() if line.startswith("№")]) == 9


def test_the_scan_summary_counts_nothing_it_cannot_judge():
    """Решение владельца 2026-09-12: счётчики покрытия из сводки убраны целиком.

    «Оценено решений: 0 из 4» и «Суммарная потеря 0.0 bb» на файле, где движок
    не судит ни одной точки, говорили только о самом движке. Пустой список
    расхождений теперь не комментируется вовсе — ни числом, ни фразой.
    """
    empty = ScanSummary(
        hands_total=4,
        hands_with_decision=0,
        items=[],
        total_loss_bb=0.0,
        points_total=4,
        points_judged=0,
        hand_nos=["H1", "H2", "H3", "H4"],
    )
    text = scan_summary_msg(empty, quota_left=1, quota_total=5).text
    assert text.startswith("Скан завершён: 4 руки.")
    for gone in ("Оценено решений", "Суммарная потеря", "Решений без оценки", "с решением"):
        assert gone not in text, f"счётчик остался в сводке: «{gone}»"
    assert "Разобрать раздачу" in text, "дверь в разбор под каждой рукой обязана остаться"


# --- разбор раздачи -------------------------------------------------------------------


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


def _deep_dive(res: AnalysisResult, en=None, **kw) -> Msg:
    """Разбор раздачи в тестах: раздача обязательна, остальное — по умолчанию.

    `hand_analysis_msgs` печатает сырые числа раздачи, и взять их неоткуда, кроме
    `EnrichedHand`; тестам, которые проверяют не числа, а статус-строку или
    кнопки, подставляется любая настоящая раздача. Возвращается ПЕРВОЕ сообщение:
    про второе (хвост чисел) говорит отдельный тест.
    """
    return _deep_dive_all(res, en, **kw)[0]


def _deep_dive_all(res: AnalysisResult, en=None, **kw) -> list[Msg]:
    """Все сообщения разбора — одно или два."""
    kw.setdefault("elapsed_s", 12)
    kw.setdefault("zone", Zone.STRICT)
    kw.setdefault("quota_left", 17)
    kw.setdefault("quota_total", 50)
    return hand_analysis_msgs(res, en if en is not None else _postflop_hand(), **kw)


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


def test_the_hand_analysis_has_status_line_with_zone_time_and_quota():
    res = _mixed_result()
    msg = _deep_dive(res)
    assert "⏱ 12с" in msg.text
    assert "зона: строго" in msg.text
    assert "разборов 17/50 за 24 ч" in msg.text


def test_the_hand_analysis_status_line_shows_assuming_zone_word():
    res = _mixed_result()
    msg = _deep_dive(res, zone=Zone.ASSUMING, quota_left=1)
    assert "зона: предполагая" in msg.text


def test_the_verdict_buttons_are_two():
    """Реплей переехал в разбор (задача 3); кнопка, показывающая его второй раз,
    осталась бы без содержания. `deep_dive_button` («разобрать» под сканом) —
    другая кнопка, её план не касается."""
    from harness.presentation import verdict_buttons

    assert [b.text for b in verdict_buttons("TM99")] == ["🎯 Диапазоны", "✋ Не согласен"]


def test_the_hand_analysis_buttons_are_ranges_and_disagree_with_hand_no():
    res = _mixed_result(hand_no="H99")
    msg = _deep_dive(res, elapsed_s=1, quota_left=1, quota_total=1)
    assert len(msg.buttons) == 1
    row = msg.buttons[0]
    assert [b.text for b in row] == ["🎯 Диапазоны", "✋ Не согласен"]
    assert [b.callback_data for b in row] == ["ranges:H99", "disagree:H99"]


def test_the_hand_analysis_dev_line_appears_only_when_passed():
    res = _mixed_result()
    without = _deep_dive(res, quota_left=1, quota_total=1)
    with_dev = _deep_dive(
        res, quota_left=1, quota_total=1, dev_line="себестоимость: $0.0042, gpt-4o-mini"
    )
    assert "себестоимость" not in without.text
    assert "$0.0042" not in without.text
    assert "себестоимость: $0.0042, gpt-4o-mini" in with_dev.text


def test_the_hand_analysis_with_no_points_at_all_still_shows_the_hand():
    res = AnalysisResult(hand_no="H0", points=[], ranked=[], total_ev_loss_bb=0.0)
    msg = _deep_dive(res, elapsed_s=3, quota_left=1, quota_total=1)
    assert "Рука H0" in msg.text
    assert "⏱ 3с" in msg.text


# --- escalation_msg ------------------------------------------------------------------


def test_escalation_msg_has_option_buttons_plus_manual_entry():
    """Кнопка несёт НОМЕР задачи и ИНДЕКС варианта, а не сам вариант.

    Номер задачи — потому что ждущих задач у игрока бывает несколько сразу, и
    ответ без него уходил бы в чужую руку. Индекс — потому что в `callback_data`
    Телеграма 64 байта, а вариантом бывает ник игрока.
    """
    msg = escalation_msg(
        77, field="hero_stack_bb", question="Стек героя: 12.7bb?", options=["12.7", "12.1"]
    )
    assert msg.text == "Стек героя: 12.7bb?"
    assert len(msg.buttons) == 1
    row = msg.buttons[0]
    assert [b.text for b in row] == ["12.7", "12.1", "ввести вручную"]
    assert [b.callback_data for b in row] == [
        "escalate:77:hero_stack_bb:0",
        "escalate:77:hero_stack_bb:1",
        "escalate:77:hero_stack_bb:manual",
    ]


def test_every_escalation_button_fits_the_telegram_callback_limit():
    """64 байта — жёсткий предел Телеграма, и ник в варианте его пробивал бы."""
    long_nicks = [f"игрок_с_очень_длинным_ником_{i}" for i in range(4)]
    msg = escalation_msg(9_999_999, field="button", question="Кто?", options=long_nicks)
    assert all(
        len(btn.callback_data.encode("utf-8")) <= 64 for row in msg.buttons for btn in row
    )


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


# --- вход игрока (задача 19) ---------------------------------------------------------


def test_entry_messages_are_plain_text_without_buttons():
    """Ни у одного сообщения входа нет кнопок: продукт обещает, что основное
    действие — не кнопка, и первый экран не начинается с меню.
    """
    for msg in (
        start_msg(),
        hh_accepted_msg(),
        hh_scan_in_progress_msg(),
        bot_failure_msg(),
        unsupported_document_msg(),
        new_session_msg("Сессия 4 сен", previous_closed=True),
    ):
        assert msg.text.strip()
        assert msg.buttons == []


def test_start_msg_does_not_deny_the_menu_it_carries():
    """Первая строка докстринга обещала «без меню», четвёртый абзац — обратное.

    Меню сообщение действительно везёт, значит неверна была первая строка.
    """
    from harness.presentation import MAIN_MENU, start_msg

    assert start_msg().menu == MAIN_MENU
    assert "без меню" not in (start_msg.__doc__ or "")


def test_hand_in_progress_msg_names_the_next_step_and_blames_nobody():
    """Отказ без следующего шага оставляет игрока с той же картинкой в руках.

    Слова про ошибку тут быть не может: экран прочитан, просто рука на нём ещё
    не доиграна, и «пришлите позже» — единственное, что игроку остаётся сделать.
    """
    from harness.presentation import hand_in_progress_msg

    msg = hand_in_progress_msg()

    assert msg.buttons == []
    assert "скриншот" in msg.text
    assert "ошиб" not in msg.text


def test_unknown_button_msg_does_not_promise_a_button_that_will_work_later():
    """Три кнопки под вердиктом разбираются по-настоящему с задачи 23.

    Текст обещал, что кнопка «появится вместе с разбором словами», а сюда
    доходит теперь только нажатие, которого не разобрал никто.
    """
    from harness.presentation import unknown_button_msg

    text = unknown_button_msg().text
    assert "появится" not in text
    assert len(text) <= 200  # предел всплывающего уведомления callback-ответа


def test_gg_nickname_too_long_msg_does_not_speak_for_the_room():
    """64 — ширина колонки `players.gg_nickname`, а не правило GG.

    Правил рума про длину ника проект не знает, а текст утверждал их игроку.
    """
    from harness.presentation import gg_nickname_too_long_msg

    text = gg_nickname_too_long_msg(64).text
    assert "в руме" not in text
    assert "64" in text


def test_hh_accepted_msg_promises_nothing_it_cannot_know():
    """Подтверждение приёма не называет ни числа рук, ни времени ожидания — файл
    ещё не разобран, и любое такое число было бы выдуманным (CLAUDE.md).

    Проверка «нет цифр» груба намеренно: она ловит и «через ~40 секунд», и «146
    раздач» — оба способа сказать то, чего бот в этот момент не знает.
    """
    assert not any(char.isdigit() for char in hh_accepted_msg().text)


def test_new_session_msg_mentions_closing_only_when_something_was_closed():
    """Первая сессия игрока: закрывать было нечего — и сообщение об этом молчит.
    Обе стороны проверяются вместе, иначе шаблон, безусловно печатающий (или
    безусловно не печатающий) фразу, прошёл бы тест случайно.
    """
    first = new_session_msg("Сессия 4 сен", previous_closed=False)
    later = new_session_msg("Сессия 4 сен", previous_closed=True)
    assert "закрыт" not in first.text
    assert "закрыт" in later.text
    assert "Сессия 4 сен" in first.text and "Сессия 4 сен" in later.text



def test_the_in_progress_refusal_asks_to_wait_and_does_not_send_anyone_to_a_new_session():
    """Отказ на время работы обязан обещать ответ, а не отправлять игрока прочь.

    Прежний текст звал `/new`, потому что отклонял и УЖЕ РАЗОБРАННЫЙ файл: без
    новой сессии тот же турнир нельзя было разобрать повторно вовсе. Теперь
    повтор законченного скана просто принимается, и звать куда-либо не за чем —
    ждать надо секунды.
    """
    text = hh_scan_in_progress_msg().text
    assert "считаю" in text
    assert "/new" not in text


def test_bot_failure_msg_says_whose_side_it_is_without_the_reason():
    """Причина сбоя — ops-данные (лог, `jobs.error`), игроку уходит только факт
    и предложение повторить. Никакого «Traceback», путей и имён исключений."""
    text = bot_failure_msg().text
    assert "нашей стороне" in text
    assert not any(word in text for word in ("Error", "Traceback", "/data", "Exception"))


# --- Форма «около нуля»: знак, интервал и потолок цены вместо отказа -----------------


def _close_call(
    *, hand_no: str = "H7", low: float = -0.3, high: float = 0.8, point: float = 0.1
) -> ScanItem:
    return ScanItem(
        hand_no=hand_no,
        hand_index=3,
        hero_class="A5s",
        spot=SpotKind.PUSHFOLD_UNOPENED,
        action_taken="fold",
        best_action="около нуля, оба варианта допустимы",
        ev_diff_bb=0.0,
        zone=Zone.ASSUMING,
        interval=EvInterval(point_bb=point, low_bb=low, high_bb=high, near_zero=True),
    )


def test_scan_summary_msg_shows_close_calls_with_interval_and_ceiling():
    """Точка «около нуля» доходит до игрока: знак, интервал и потолок цены.

    Прежде такая точка не показывалась вообще — её снимал отказ «одного
    надёжного числа здесь нет», и вместе с числом пропадали знак, порядок
    величины и потолок. Строка сводки обязана нести все три.
    """
    s = ScanSummary(
        hands_total=10,
        hands_with_decision=8,
        items=[],
        close_calls=[_close_call()],
        total_loss_bb=0.0,
    )
    msg = scan_summary_msg(s, quota_left=17, quota_total=50)

    line = next(line for line in msg.text.splitlines() if "H7" in line)
    assert "около нуля" in line
    assert "−0.3" in line and "0.8 bb" in line  # интервал целиком
    assert "не больше 0.8 bb" in line  # потолок цены
    assert "по модели диапазонов" in line  # зона `assuming` подписана, как и у расхождений


def test_scan_summary_msg_close_calls_are_not_counted_as_discrepancies():
    """Точка «около нуля» — не расхождение: в топ расхождений она не попадает.

    Оба варианта допустимы, упрёка нет, цена ноль — поставить такую строку в
    список расхождений значило бы обвинить игрока в решении, которое сам расчёт
    считает допустимым.
    """
    s = ScanSummary(
        hands_total=10,
        hands_with_decision=8,
        items=[_scan_item(hand_no="H1", ev_diff_bb=-2.3, zone=Zone.STRICT)],
        close_calls=[_close_call(hand_no="H7")],
        total_loss_bb=-2.3,
    )
    msg = scan_summary_msg(s, quota_left=1, quota_total=1)

    lines = msg.text.splitlines()
    top_at = next(i for i, line in enumerate(lines) if "Топ расхождений" in line)
    close_at = next(i for i, line in enumerate(lines) if "около нуля" in line and "H7" in line)
    assert top_at < close_at  # раздел «около нуля» идёт ПОСЛЕ расхождений
    assert next(i for i, line in enumerate(lines) if "H1" in line) < close_at
    # Кнопка разбора стоит только под расхождением: у точки «около нуля» разбирать нечего.
    assert len(msg.buttons) == 1


def test_scan_summary_msg_without_close_calls_says_nothing_about_them():
    """Раздела нет, когда точек «около нуля» нет: сообщение о том, чего не было."""
    s = ScanSummary(hands_total=5, hands_with_decision=5, items=[], total_loss_bb=0.0)
    assert "около нуля" not in scan_summary_msg(s, quota_left=1, quota_total=1).text


def test_scan_summary_msg_says_which_end_of_the_close_calls_it_kept():
    """Обрезанный список «около нуля» называет, по какому краю он обрезан.

    Строка обещает игроку, что выбор дёшев, и список идёт от самого дешёвого
    выбора к самому дорогому (`scan.scan_tournament`). Подпись обязана говорить
    то же самое: прежняя («дороже — первыми») описывала порядок, которого больше
    нет, и обещала игроку ровно обратное тому, что он увидит.
    """
    s = ScanSummary(
        hands_total=30,
        hands_with_decision=30,
        items=[],
        close_calls=[_close_call(hand_no=f"H{i}") for i in range(26)],
        total_loss_bb=0.0,
    )
    head = next(
        line for line in scan_summary_msg(s, 1, 1).text.splitlines() if "около нуля" in line
    )
    assert "показаны 10 из 26" in head
    assert "самые дешёвые — первыми" in head


def test_the_close_call_line_agrees_with_itself_after_rounding():
    """Показанный интервал обязан содержать показанный потолок, а не спорить с ним.

    При округлении концов к ближайшему интервал −3.34 … +2.61 печатался как
    «от −3.3 до +2.6», а потолок (округляемый вверх) — как «не больше 3.4»:
    числа рядом, и большее из них в интервале не видно. Концы округляются
    наружу, поэтому потолок всегда равен модулю худшего из ПОКАЗАННЫХ концов.
    """
    s = ScanSummary(
        hands_total=1,
        hands_with_decision=1,
        items=[],
        close_calls=[_close_call(low=-3.34, high=2.61, point=-0.4)],
        total_loss_bb=0.0,
    )
    line = next(line for line in scan_summary_msg(s, 1, 1).text.splitlines() if "H7" in line)
    assert "от −3.4 bb до +2.7 bb" in line
    assert "не больше 3.4 bb" in line


# --- блок «Что было» и бюджет сообщения ----------------------------------------------


def _two_point_result() -> AnalysisResult:
    """Две точки с РАЗНЫМИ `dp_index` — как их отдаёт ядро на настоящей раздаче."""
    strict_point = _point(spot=SpotKind.PUSHFOLD_UNOPENED, ev_diff_bb=-2.3, zone=Zone.STRICT)
    assuming_point = _point(
        spot=SpotKind.PUSHFOLD_FACING_SHOVE,
        ev_diff_bb=-1.1,
        zone=Zone.ASSUMING,
        assumption=Assumption(range=Range(weights={"AA": 1.0}), source="model:test"),
    ).model_copy(update={"dp_index": 3})
    return AnalysisResult(
        hand_no="H42",
        points=[strict_point, assuming_point],
        ranked=[0, 1],
        total_ev_loss_bb=-3.4,
    )


def _padded_result(filler: str) -> AnalysisResult:
    """Тот же разбор, но с сырыми числами, которые заведомо не влезают в 4096.

    Длина берётся из `detail`, а не из выдуманного поля: это и есть содержимое,
    которое режется в тесноте (`_shrink_rows`).
    """
    res = _two_point_result()
    points = [
        point.model_copy(update={"detail": {"zone_reason": filler * 3000}})
        for point in res.points
    ]
    return res.model_copy(update={"points": points})


def _replay() -> HandReplay:
    return HandReplay(
        spans=[
            ReplaySpan(text="Вы на SB, J♥️9♥️, 10.0 ББ.\nUTG фолд → "),
            ReplaySpan(text="вы олл-ин 9.9", emphasis=True),
            ReplaySpan(text=" → BB & CO фолд."),
        ]
    )


def test_the_deep_dive_opens_with_what_happened_in_html():
    """Блок «Что было» — первым в разборе (спека §5.6, план 2026-09-12).

    Выделение точки решения — разметка Телеграма, поэтому у сообщения стоит
    `parse_mode`, а весь остальной текст экранируется.
    """
    msg = _deep_dive(_two_point_result(), replay=_replay())
    assert msg.parse_mode == "HTML"
    assert msg.text.startswith("Что было\n")
    assert "<b>вы олл-ин 9.9</b>" in msg.text
    assert "<b>Вы на SB" not in msg.text, "жирным — только помеченный кусок"
    assert "BB &amp; CO" in msg.text, "остальной текст экранируется"


def test_without_a_replay_the_deep_dive_stays_plain():
    """Без блока разметки в сообщении нет — и экранировать текст незачем."""
    msg = _deep_dive(_two_point_result())
    assert msg.parse_mode is None and "Что было" not in msg.text


def test_a_hand_that_fits_goes_out_in_one_message():
    en = _postflop_hand()
    msgs = _deep_dive_all(analyze_hand(en), en, replay=_replay())
    assert len(msgs) == 1
    assert len(msgs[0].text) <= 4096


def test_a_hand_too_long_for_one_message_goes_out_in_two():
    """Резать разбор внутри точки нельзя, выбрасывать посчитанное — тоже: хвост
    уезжает вторым сообщением, начинаясь с целой точки.

    Реплей растянут искусственно, потому что настоящий в предел укладывается:
    проверяется поведение при переполнении, а не длина конкретной раздачи.
    """
    from tests.test_river_analysis import _river_hand

    en = _river_hand()
    res = analyze_hand(en)
    assert len(res.points) == 7, "фикстура перестала быть семиточечной"
    long_replay = HandReplay(spans=[ReplaySpan(text="а" * 1500), ReplaySpan(text="шов", emphasis=True)])

    msgs = _deep_dive_all(res, en, replay=long_replay)
    assert len(msgs) == 2
    first, second = msgs
    assert len(first.text) <= 4096 and len(second.text) <= 4096
    assert first.text.startswith("Что было\n")
    assert "Сверка денег с источником" in first.text, "сверка несжимаема и живёт в первом"
    assert "разборов 17/50" in first.text, "статус-строка не уезжает"
    assert first.buttons and not second.buttons
    assert re.match(r"^\d+\. (префлоп|флоп|тёрн|ривер)", second.text), (
        "второе сообщение начинается с целой точки"
    )
    assert "\n\n\n" not in first.text


def test_the_two_messages_together_carry_every_point_of_the_hand():
    from tests.test_river_analysis import _river_hand

    en = _river_hand()
    res = analyze_hand(en)
    long_replay = HandReplay(spans=[ReplaySpan(text="а" * 1500)])
    whole = "\n".join(msg.text for msg in _deep_dive_all(res, en, replay=long_replay))
    for point in res.points:
        assert f"{point.dp_index + 1}. " in whole, f"точка {point.dp_index} потерялась"


def test_the_deep_dive_escapes_a_nickname_that_looks_like_a_tag():
    """Ники приходят со скрина: незакрытый `<` уронил бы отправку целиком."""
    replay = HandReplay(
        spans=[ReplaySpan(text="<script> & Hero"), ReplaySpan(text="шов", emphasis=True)]
    )
    msg = _deep_dive(_two_point_result(), replay=replay)
    assert "&lt;script&gt; &amp; Hero" in msg.text
    assert "<script>" not in msg.text


def test_a_real_hand_analysis_fits_one_telegram_message():
    en = _postflop_hand()
    msg = _deep_dive(analyze_hand(en), en, replay=_replay())
    assert len(msg.text) < 4096


def test_the_vision_messages_speak_the_product_voice_and_carry_no_buttons():
    """Те же два требования, что ко всем текстам входа: не пусто и без кнопок.

    Кнопки на этом пути есть ровно у одного сообщения — вопроса эскалации, и
    собирает их `escalation_buttons`, а не конструктор текста.
    """
    from harness.presentation import (
        ask_gg_nickname_msg,
        gg_nickname_saved_msg,
        not_a_hand_msg,
        send_as_file_msg,
        vision_answer_not_a_number_msg,
        vision_answer_saved_msg,
        vision_gave_up_msg,
        vision_manual_entry_msg,
    )

    for msg in (
        ask_gg_nickname_msg(),
        gg_nickname_saved_msg("nick"),
        not_a_hand_msg("это лобби турнира"),
        send_as_file_msg(),
        vision_manual_entry_msg("Банк распознан верно?"),
        vision_answer_saved_msg(),
        vision_answer_not_a_number_msg(),
        vision_gave_up_msg(),
    ):
        assert msg.text.strip()
        assert msg.buttons == []


def test_the_file_hint_appears_only_where_it_is_earned_and_explains_the_gesture():
    """Просьба переслать файлом обязана сказать, ГДЕ в Телеграме этот выбор.

    Трение вводится только после несошедшейся проверки карт (сжатие безопасно
    для чисел и опасно для мелких значков мастей), и просьба без инструкции
    стоила бы игроку того же тупика, что отказ без пути дальше.
    """
    from harness.presentation import send_as_file_msg

    text = send_as_file_msg().text
    assert "файл" in text.casefold()
    assert "скрепк" in text.casefold() or "документ" in text.casefold()


def test_the_escalation_message_offers_both_readings_plus_manual_entry():
    """Варианты — два независимых прочтения одного экрана, а не выдуманные числа."""
    from harness.presentation import escalation_msg

    msg = escalation_msg(1, "pot", "Банк распознан верно?", ["31.95", "30.74"])
    labels = [btn.text for row in msg.buttons for btn in row]
    assert labels == ["31.95", "30.74", "ввести вручную"]


def test_the_refusal_repeats_what_the_model_saw_instead_of_a_generic_phrase():
    """Модель видела экран, а мы нет: её причина — единственная подсказка игроку."""
    from harness.presentation import not_a_hand_msg

    assert "список результатов" in not_a_hand_msg("список результатов").text


def test_what_could_not_be_checked_reaches_the_player_and_kills_the_strict_badge():
    """«Проверить было нечем» — не то же, что «проверено и сошлось» (ревью раунда 1, C).

    Скрин с фабрикованным шоудауном показывал вскрытие, которого никто не видел,
    и подписывался «зона: строго» — ровно то, что пометка обещала не допустить.
    Проверяются обе половины сразу: названная оговорка в тексте и отсутствие
    самой уверенной подписи продукта.
    """
    res = AnalysisResult(
        hand_no="TM1",
        points=[_point(spot=SpotKind.PUSHFOLD_UNOPENED, ev_diff_bb=-1.0, zone=Zone.STRICT)],
        ranked=[0],
    )
    plain = _deep_dive(res)
    caveated = _deep_dive(
        res, zone=Zone.ASSUMING, not_checked=["сверка получателей банка"]
    )
    assert "Проверить на этом экране было нечем: сверка получателей банка." in caveated.text
    assert "Проверить на этом экране было нечем" not in plain.text
    assert "зона: строго" in plain.text
    assert "зона: строго" not in caveated.text


def test_an_unjudged_point_shows_the_reason_the_core_recorded():
    """Решение владельца 2026-09-12: `detail` печатается целиком, с подписью.

    Прежде причина отказа ядра игроку не показывалась вовсе, и точка без
    вердикта была молчанием; теперь молчания нет — есть подписанное «почему
    вердикта нет».
    """
    unpriced = _point(
        spot=SpotKind.PREFLOP_OTHER, ev_diff_bb=0.0, zone=Zone.STRICT
    ).model_copy(update={"best_action": "", "detail": {"unjudged": "перебор подмножеств"}})
    msg = _deep_dive(
        AnalysisResult(hand_no="TM1", points=[unpriced], ranked=[]), elapsed_s=5, zone=None
    )
    assert "    вердикта нет: перебор подмножеств" in msg.text
    assert "точек с вердиктом нет" not in msg.text


# --- риверная точка: числа без цены ---------------------------------------------------

# Живая рука: борд `Jc 6d As 2h Ac`, у героя `Jh Ts`. Числа посчитаны
# инструментом и проверены в `test_river_call`/`test_river_analysis`; здесь они
# заданы вручную, потому что проверяется РЕНДЕР, а не расчёт.
_LIVE_RIVER = {
    "pot_before": 398_000,
    "to_call": 169_000,
    "required_equity": 0.2980599647266314,
    "min_value_combos": 94,
    "bluffs_needed_min_value": 38.18844221105527,
    "bluff_share": 0.28889395753739705,
    "fold_proven": False,
}


def _river_point(*, best_action: str = "", **over) -> PointVerdict:
    """Точка ривера с посчитанными числами: `best_action` пуст, пока фолд не доказан."""
    from harness.contracts import RIVER_CALL_DETAIL

    return PointVerdict(
        dp_index=6,
        street=Street.RIVER,
        spot=SpotKind.POSTFLOP,
        zone=Zone.STRICT,
        action_taken="call",
        best_action=best_action,
        ev_diff_bb=0.0,
        detail={RIVER_CALL_DETAIL: {**_LIVE_RIVER, **over}},
    )


def _point_block(text: str, head: str) -> str:
    """Строки одной постфлоп-точки: от её заголовка до статус-строки разбора."""
    block = text[text.index(head) :]
    return block.split("\n⏱", 1)[0]


def _one_point_msg(point: PointVerdict, zone: Zone | None = None) -> str:
    """Разбор из одной постфлоп-точки — ривера, тёрна или флопа."""
    return _deep_dive(
        AnalysisResult(hand_no="H7", points=[point], ranked=[]), zone=zone
    ).text


def test_the_river_line_shows_the_bluff_requirement():
    """Форма, согласованная с владельцем, — дословно.

    Банка, доплаты и требуемой эквити здесь нет: их печатает строка шансов банка
    той же точки, и второй раз те же три числа были бы дублем.
    """
    text = _one_point_msg(_river_point())
    assert (
        "    Чтобы колл вышел в ноль, на 94 комбинации несомненного вэлью ему нужно "
        "38 блефов — то есть блефом должно быть 28.9% его ставящего диапазона."
    ) in text
    assert "Ривер: банк" not in text, "цена колла живёт в строке шансов банка"
    assert "к оплате" not in text


def test_a_river_point_alone_is_not_a_hand_without_a_verdict():
    """Точка без цены, но с числами, перестала быть молчанием."""
    text = _one_point_msg(_river_point())
    assert "точек с вердиктом нет" not in text
    assert "несомненного вэлью" in text


def test_a_proven_fold_names_the_line_and_its_single_assumption():
    """Доказанный фолд: лучшая линия названа, и допущение под ней — тоже."""
    text = _one_point_msg(_river_point(best_action="fold", fold_proven=True), Zone.STRICT)
    assert "больше, чем на этом борде существует." in text
    assert "    Лучше: фолд. Допущение: сильнейшие руки он ставит." in text
    assert "зона: строго" in text
    # Доля блефов в диапазоне на доказанном фолде не печатается: диапазона,
    # в котором она бы считалась, на этом борде не существует.
    assert "ставящего диапазона" not in text


def test_a_proven_fold_names_the_line_even_without_the_bluff_line():
    """Вырожденное требование не должно уносить с собой вывод."""
    text = _one_point_msg(
        _river_point(
            best_action="fold", fold_proven=True, min_value_combos=0, bluffs_needed_min_value=0.0
        )
    )
    assert "Чтобы колл вышел в ноль" not in text
    assert "Лучше: фолд." in text


def test_a_degenerate_river_requirement_prints_no_requirement_at_all():
    """Ни вэлью старшего класса, ни блефов — «нужно 0 блефов» не утверждение."""
    text = _one_point_msg(_river_point(min_value_combos=0, bluffs_needed_min_value=0.0))
    assert "блеф" not in text
    assert "Чтобы колл вышел в ноль" not in text


def test_half_a_bluff_is_rounded_up_to_one():
    """Требование читается в штуках, и половина обязана дать один, а не ноль."""
    text = _one_point_msg(_river_point(min_value_combos=1, bluffs_needed_min_value=0.5))
    assert "на 1 комбинацию несомненного вэлью ему нужен 1 блеф" in text


def test_the_bluff_count_agrees_with_its_own_plural_form():
    """Форма слова идёт за числом, а не за самым частым случаем."""
    two = _one_point_msg(_river_point(min_value_combos=22, bluffs_needed_min_value=2.0))
    assert "на 22 комбинации несомненного вэлью ему нужно 2 блефа" in two
    many = _one_point_msg(_river_point(min_value_combos=15, bluffs_needed_min_value=11.0))
    assert "на 15 комбинаций несомненного вэлью ему нужно 11 блефов" in many


def test_the_river_block_shows_neither_the_enumeration_nor_the_missing_proof():
    """Двух вещей в разборе ривера нет: перебора комбо и рассказа о недоказанном.

    Перебор не показывается тем, что его НЕТ В КОНТРАКТЕ: числа исходов
    (сколько комбо бьёт, проигрывает, делит) до изложения не доезжают вовсе, и
    напечатать их ему не из чего.
    """
    from harness.contracts import RiverCallDetail
    from harness.explanation.faithfulness import error_words_in

    fields = set(RiverCallDetail.model_fields)
    assert not fields & {"combos_total", "combos_ahead", "combos_behind", "combos_tied"}

    # Проверяется блок САМОЙ точки, а не всё сообщение: шапка раздачи называет
    # улицы борда и банка по праву — это числа раздачи, а не рассказ о ривере.
    block = _point_block(_one_point_msg(_river_point()), "Чтобы колл вышел в ноль")
    for forbidden in ("доказать", "не удалось", "тёрн", "флоп", "990", "862"):
        assert forbidden not in block.lower()
    assert error_words_in(block) == []


# --- точка тёрна и флопа: те же слова, что у ривера ----------------------------------

# Живая рука на тёрне: борд `Jc 6d As 2h`, у героя `Jh Ts`. Числа посчитаны
# инструментом и проверены в `test_turn_flop_call`/`test_turn_flop_analysis`;
# здесь они заданы вручную, потому что проверяется РЕНДЕР, а не расчёт.
_LIVE_TURN = {
    "pot_before": 160_000,
    "to_call": 69_000,
    "required_equity": 0.30131004366812225,
    "min_value_combos": 55,
    "bluffs_needed_min_value": 18,
    "bluff_share": 18 / 73,
}


def _turn_point(*, street: Street = Street.TURN, **over) -> PointVerdict:
    """Точка тёрна или флопа с посчитанными числами: `best_action` всегда пуст."""
    from harness.contracts import TURN_FLOP_CALL_DETAIL

    return PointVerdict(
        dp_index=4,
        street=street,
        spot=SpotKind.POSTFLOP,
        zone=Zone.STRICT,
        action_taken="call",
        best_action="",
        ev_diff_bb=0.0,
        detail={TURN_FLOP_CALL_DETAIL: {**_LIVE_TURN, **over}},
    )


def test_the_turn_line_is_worded_exactly_like_the_river_line():
    """Подписи у чисел одни и те же на обеих улицах."""
    text = _one_point_msg(_turn_point())
    assert (
        "    Чтобы колл вышел в ноль, на 55 комбинаций несомненного вэлью ему нужно "
        "18 блефов — то есть блефом должно быть 24.7% его ставящего диапазона."
    ) in text


def test_the_street_of_a_point_is_taken_from_its_own_data():
    """Улица берётся у точки, а не прибита к риверу."""
    text = _one_point_msg(_turn_point(street=Street.FLOP))
    assert "5. флоп" in text
    assert "ривер" not in text and "тёрн" not in text


def test_a_requirement_beyond_the_board_prints_no_number_of_bluffs():
    """Числа блефов не существует — строка называет вэлью и говорит это словами."""
    text = _one_point_msg(_turn_point(bluffs_needed_min_value=None, bluff_share=None))
    assert (
        "    Чтобы колл вышел в ноль, на 55 комбинаций несомненного вэлью ему нужно "
        "больше блефов, чем на этом борде существует."
    ) in text
    assert "ставящего диапазона" not in text


def test_a_turn_point_names_no_better_line_and_no_reservations():
    """Лучшей линии на этих улицах нет, и рассказа о том, чего нет, — тоже."""
    from harness.explanation.faithfulness import error_words_in

    block = _point_block(_one_point_msg(_turn_point()), "Чтобы колл вышел в ноль")
    assert "Лучше:" not in block
    assert "Допущение" not in block
    assert "точек с вердиктом нет" not in block
    for forbidden in ("доказать", "не удалось", "ривер", "990", "1035"):
        assert forbidden not in block.lower()
    assert error_words_in(block) == []


def test_the_streets_are_printed_in_the_order_they_were_dealt():
    """Флоп, тёрн, ривер — в порядке раздачи, а не в порядке появления расчётов."""
    flop = _turn_point(street=Street.FLOP).model_copy(update={"dp_index": 3})
    text = _deep_dive(
        AnalysisResult(hand_no="H8", points=[flop, _turn_point(), _river_point()], ranked=[]),
        zone=None,
    ).text
    assert text.index("4. флоп") < text.index("5. тёрн") < text.index("7. ривер")


# --- экраны нижнего меню (задача 23) -------------------------------------------------


def _leak_overview(*pairs, judged: int = 10, total: int = 40):
    """Экран «Мои лики» из пар «ключ правила → (частота, цена)»."""
    from harness.contracts import LEAK_RULES, LeaksOverview, LeakStat

    by_key = {rule.key: rule for rule in LEAK_RULES}
    return LeaksOverview(
        points_judged=judged,
        points_total=total,
        leaks=[
            LeakStat(rule=by_key[key], count=count, loss_bb=loss)
            for key, count, loss in pairs
        ],
    )


def test_leaks_msg_names_frequency_and_cost_next_to_every_type():
    """Рядом с каждым типом лика — частота и цена (постановка владельца дословно)."""
    from harness.presentation import leaks_msg

    msg = leaks_msg(_leak_overview(("no_shove", 7, 5.4), ("call_too_wide", 1, 3.1)))

    assert "Не шовит, где надо — 7 раз, −5.4 bb" in msg.text
    assert "Коллирует шов слишком широко — 1 раз, −3.1 bb" in msg.text


def test_leaks_msg_puts_the_coverage_above_the_list():
    """Покрытие — первым: короткий список ликов без него читается как «сыграно чисто»."""
    from harness.presentation import leaks_msg

    msg = leaks_msg(_leak_overview(("no_shove", 2, 1.0), judged=10, total=40))

    assert "Оценено решений: 10 из 40 за всю историю." in msg.text
    assert msg.text.index("Оценено решений") < msg.text.index("Не шовит")


def test_leaks_msg_of_a_clean_history_still_shows_the_coverage():
    """Пустой список — не «всё хорошо»: покрытие и оговорка остаются на месте."""
    from harness.presentation import leaks_msg

    msg = leaks_msg(_leak_overview(judged=3, total=41))

    assert "не нашлось" in msg.text
    assert "Оценено решений: 3 из 41" in msg.text
    assert "не судит" in msg.text


def test_leaks_msg_without_any_analysis_says_there_is_nothing_to_count():
    """Ноль точек — это «не из чего считать», а не «ликов нет»."""
    from harness.presentation import leaks_msg

    msg = leaks_msg(_leak_overview(judged=0, total=0))

    assert "не из чего считать" in msg.text.lower()
    assert "Оценено решений" not in msg.text


def test_leaks_msg_never_blames_the_outcome_of_a_hand():
    """Лик — про решение против диапазона, а не про проигранную раздачу (CLAUDE.md)."""
    from harness.presentation import leaks_msg

    msg = leaks_msg(_leak_overview(("no_shove", 3, 2.0)))

    assert "ошиб" not in msg.text.lower()
    assert "по решениям, а не по исходам" in msg.text


def _session_line(session_id: int, title: str, *, active: bool):
    from datetime import UTC, datetime

    from harness.contracts import SessionLine

    return SessionLine(
        session_id=session_id, title=title, started_at=datetime.now(UTC), is_active=active
    )


def test_sessions_msg_marks_the_open_evening_and_offers_a_new_one():
    from harness.presentation import sessions_msg

    msg = sessions_msg(
        [
            _session_line(2, "Сессия 7 сен", active=True),
            _session_line(1, "Сессия 5 сен", active=False),
        ],
        2,
    )

    assert "Сессия 7 сен · сейчас идёт" in msg.text
    assert "Сессия 5 сен\n" in msg.text
    data = [btn.callback_data for row in msg.buttons for btn in row]
    assert data == ["session:2", "session:1", "newsession"]


def test_sessions_msg_without_a_single_session_still_offers_to_start_one():
    from harness.presentation import sessions_msg

    msg = sessions_msg([], 0)

    assert "Сессий пока нет" in msg.text
    assert [btn.callback_data for row in msg.buttons for btn in row] == ["newsession"]


def test_sessions_msg_says_when_the_history_did_not_fit():
    """Экран отдаёт страницу истории и обязан сказать, что она страница.

    Молчаливая обрезка запрещена этим же файлом дважды — в сводке скана и в
    отчёте по турниру.
    """
    from harness.presentation import sessions_msg

    lines = [_session_line(i, f"Сессия {i}", active=False) for i in range(10, 0, -1)]

    msg = sessions_msg(lines, 34)

    assert "Показаны 10 из 34 — самые свежие." in msg.text


def _summary(**over):
    from harness.contracts import SessionSummary

    base = {
        "session_id": 1,
        "title": "Сессия 5 сен",
        "tournaments": 1,
        "hands": 12,
        "loss_bb": 6.3,
        "points_judged": 18,
        "points_total": 24,
    }
    base.update(over)
    return SessionSummary(**base)


def test_session_summary_msg_counts_the_evening_and_names_its_leak():
    from harness.contracts import LEAK_RULES, LeakStat
    from harness.presentation import session_summary_msg

    rule = next(r for r in LEAK_RULES if r.key == "fold_vs_shove")
    msg = session_summary_msg(
        _summary(top_leak=LeakStat(rule=rule, count=3, loss_bb=4.1))
    )

    assert "Турниров: 1 · разобрано раздач: 12." in msg.text
    assert "Оценено решений: 18 из 24 за этот вечер." in msg.text
    assert "Суммарная потеря в оценённых решениях: −6.3 bb." in msg.text
    assert "Сбрасывает против шова, где колл плюсовой — 3 раза, −4.1 bb" in msg.text


def test_session_summary_msg_scopes_the_loss_to_the_judged_points():
    """Сводка вечера подписывает сумму тем же множеством, что и сводка скана.

    `SessionsRepo._session_loss_bb` складывает отрицательные `ev_diff_bb` только
    по судимым точкам, поэтому «по всем точкам разбора» рядом со строкой
    покрытия, которая сама говорит `18 из 24`, утверждало бы, что в сумму вошли
    и шесть точек без вердикта.
    """
    from harness.presentation import session_summary_msg

    msg = session_summary_msg(_summary(loss_bb=0.0, points_judged=18, points_total=24))

    assert "Оценено решений: 18 из 24 за этот вечер." in msg.text
    assert "Суммарная потеря в оценённых решениях: 0.0 bb." in msg.text
    assert "по всем точкам разбора" not in msg.text


def test_session_summary_msg_signs_the_leak_by_its_price_not_its_frequency():
    """Подпись обязана называть то, что посчитано: `top_leak` — самый ДОРОГОЙ.

    Список упорядочен ценой (`LeaksRepo.by_type`), поэтому первым встаёт лик,
    который случился реже дешёвого. Подпись «чаще всего» под этой строкой была
    бы утверждением о частоте, которого никто не считал.
    """
    from harness.contracts import LEAK_RULES, LeakStat
    from harness.presentation import session_summary_msg

    rule = next(r for r in LEAK_RULES if r.key == "fold_vs_shove")
    msg = session_summary_msg(_summary(top_leak=LeakStat(rule=rule, count=2, loss_bb=20.0)))

    assert "Чаще всего" not in msg.text
    assert "Дороже всего за вечер:" in msg.text


def test_session_summary_msg_stays_silent_about_a_leak_it_did_not_find():
    """Безусловная строка «повторяющийся лик» была бы сообщением о ненайденном."""
    from harness.presentation import session_summary_msg

    msg = session_summary_msg(_summary(top_leak=None))

    assert "не нашёл" in msg.text


def test_session_summary_msg_of_an_empty_evening_promises_no_numbers():
    from harness.presentation import session_summary_msg

    msg = session_summary_msg(_summary(hands=0, tournaments=0, loss_bb=0.0))

    assert "ещё ничего не разобрано" in msg.text
    assert "Суммарная потеря" not in msg.text


def _canonical_with_nicks(*nicks: str):
    """Каноническая рука со скрина: герой плюс названные ником оппоненты."""
    from harness.contracts import Provenance, RawHand
    from harness.normalizer import normalize
    from tests.test_contracts import make_min_raw

    seats = [{"seat": 4, "label": "Hero", "stack": 100_000}]
    nicknames = {"Hero": "me"}
    for index, nick in enumerate(nicks):
        seats.append({"seat": 5 + index, "label": nick, "stack": 100_000})
        nicknames[nick] = nick
    raw = RawHand.model_validate(
        make_min_raw(
            provenance=Provenance.SCREENSHOT.value,
            button_seat=4,
            seats=seats,
            vision={"displayed_pot": 100_000, "nicknames": nicknames},
        )
    )
    return normalize(raw)


def test_a_note_button_survives_a_nick_too_long_for_callback_data():
    """`callback_data` Телеграма — 64 байта, и отказ Bot API валит ВСЁ сообщение.

    Ник приезжает из чтения скрина и ничем не ограничен: кириллический ник в
    тридцать знаков — уже 60 байт, а с префиксом больше 64. Кнопка поэтому
    возит индекс, как и эскалация.
    """
    from harness.presentation import note_buttons_for_hand, note_nicks_for_hand

    long_nick = "оппонентсдлиннымименем" * 3
    hand = _canonical_with_nicks(long_nick, "villain")
    nicks = note_nicks_for_hand(hand)

    rows = note_buttons_for_hand("RC1234", nicks)

    data = [btn.callback_data for row in rows for btn in row]
    assert all(len(item.encode()) <= 64 for item in data), data
    assert data == ["note:RC1234:0", "note:RC1234:1"]
    assert nicks[0] == long_nick


def test_note_nicks_for_hand_leaves_the_hero_out():
    """Заметки пишут на других: герой — `Identity.HERO`, не `Identity.NICK`."""
    from harness.presentation import note_nicks_for_hand

    assert note_nicks_for_hand(_canonical_with_nicks("villain", "fish")) == ["villain", "fish"]


def _note(note_id: int = 1, *, nick: str = "villain", color: str = "red", text: str = "фолдит на опен"):
    from datetime import UTC, datetime

    from harness.contracts import NoteRecord

    return NoteRecord(
        note_id=note_id, nick=nick, color=color, text=text, updated_at=datetime.now(UTC)
    )


def test_notes_msg_shows_the_colour_the_nick_and_the_observation():
    from harness.presentation import notes_msg

    msg = notes_msg([_note()], 1)

    assert "🔴 агрессор · villain" in msg.text
    assert "фолдит на опен" in msg.text
    assert [btn.callback_data for btn in msg.buttons[0]] == [
        "noteedit:1",
        "notecolor:1",
        "notedel:1",
    ]


def test_notes_msg_says_where_a_new_note_starts():
    """Экран правит, но не заводит: путь заметки начинается из разбора руки."""
    from harness.presentation import notes_msg

    empty = notes_msg([], 0)
    filled = notes_msg([_note()], 1)

    for msg in (empty, filled):
        assert "из разбора раздачи" in msg.text


def test_notes_msg_of_the_longest_notes_still_fits_one_telegram_message():
    """Экран «Заметки» обязан открываться при любом наборе заметок.

    `sendMessage` отказывает на 4096 символах, а на этом же экране живут кнопки
    правки и удаления: не открывшись, он не оставляет игроку выхода — убрать
    заметку, из-за которой он не открывается, больше неоткуда.
    """
    from harness.contracts import MAX_NOTE_TEXT_CHARS
    from harness.presentation import notes_msg

    long_notes = [
        _note(note_id=i, nick=f"opponent_{i}" * 3, text="ы" * MAX_NOTE_TEXT_CHARS)
        for i in range(1, 41)
    ]

    msg = notes_msg(long_notes, len(long_notes))

    assert len(msg.text) <= 4096
    assert len(msg.buttons) < len(long_notes)
    assert f"из {len(long_notes)} — самые свежие" in msg.text


def test_notes_msg_counts_the_notes_a_player_has_not_the_page_it_was_given():
    """Знаменатель — счёт заметок игрока, а не длина отданной страницы.

    Раньше в нём стоял `limit` запроса: «Показаны 10 из 50» видел и тот, у кого
    заметок одиннадцать.
    """
    from harness.presentation import notes_msg

    notes = [_note(note_id=i, nick=f"opp{i}") for i in range(1, 21)]

    msg = notes_msg(notes[:10], 11)

    assert "из 11 — самые свежие" in msg.text
    assert "из 20" not in msg.text


def test_notes_msg_says_when_a_note_did_not_fit_whole():
    """Заметка сверх потолка обрезается с оговоркой, а не молча."""
    from harness.contracts import MAX_NOTE_TEXT_CHARS
    from harness.presentation import notes_msg

    msg = notes_msg([_note(text="я" * (MAX_NOTE_TEXT_CHARS + 500))], 1)

    assert "показано не целиком" in msg.text
    assert len(msg.text) <= 4096


def test_note_prompt_msg_of_a_long_note_still_fits_one_telegram_message():
    """Экран правки — единственный способ заменить длинную заметку."""
    from harness.presentation import note_prompt_msg

    msg = note_prompt_msg("villain", _note(text="я" * 9000))

    assert len(msg.text) <= 4096
    assert "показано не целиком" in msg.text
    assert "Новый текст заменит прежний" in msg.text


def test_note_prompt_msg_shows_what_is_already_written():
    from harness.presentation import note_prompt_msg

    fresh = note_prompt_msg("villain")
    editing = note_prompt_msg("villain", _note(text="донкает флоп"))

    assert "Сейчас записано" not in fresh.text
    assert "Сейчас записано: донкает флоп" in editing.text


def test_settings_msg_shows_the_nickname_the_quota_and_a_button_to_change_it():
    from harness.presentation import settings_msg

    known = settings_msg("screen_nick", 17, 50)
    unknown = settings_msg(None, 17, 50)

    assert "Ник в руме: screen_nick" in known.text
    assert "разборов 17/50 за 24 ч" in known.text
    assert known.buttons[0][0].callback_data == "setnick"
    assert "не задан" in unknown.text
    assert unknown.buttons[0][0].text == "✏️ Указать ник"


def test_settings_msg_mentions_the_invite_command_only_to_its_owner():
    """`/invite` — команда владельца; обычному игроку она не показывается."""
    from harness.presentation import settings_msg

    assert "/invite" in settings_msg("nick", 1, 50, is_dev=True).text
    assert "/invite" not in settings_msg("nick", 1, 50, is_dev=False).text


def test_owner_admitted_msg_does_not_claim_a_code_was_used():
    """Владельца впустило окружение сервера, а не код, — и текст это говорит.

    Отдельным конструктором, а не `invite_accepted_msg`: тот начинается словами
    «Код принят», которых на этом пути не было. Первый экран продукта не имеет
    права начинаться с неправды даже в одном слове.
    """
    from harness.presentation import invite_accepted_msg, owner_admitted_msg, start_msg

    msg = owner_admitted_msg()

    assert "Код принят" not in msg.text
    assert msg.text != invite_accepted_msg().text
    assert start_msg().text in msg.text  # тот же первый экран, а не второе приветствие
    assert "/invite" in msg.text  # единственное, что владелец умеет и чего не умеет гость
    assert msg.menu == start_msg().menu


def test_help_msg_names_the_question_command():
    """Вход в расчёты по вопросу — командой, и help обязан её назвать: без этого
    единственный способ узнать о ней — прочитать исходник."""
    from harness.presentation import help_msg

    assert "/ask" in help_msg().text


def test_help_msg_carries_the_bottom_menu_and_names_every_button():
    from harness.presentation import MAIN_MENU, help_msg

    msg = help_msg()

    assert msg.menu == MAIN_MENU
    for row in MAIN_MENU:
        for label in row:
            if label != "❓ Help":
                assert label in msg.text


def test_a_message_cannot_carry_both_keyboards():
    """У сообщения Bot API ровно одно `reply_markup` — тип не даёт собрать два."""
    import pytest

    from harness.presentation import MAIN_MENU, Btn, Msg

    with pytest.raises(ValueError, match="одна клавиатура"):
        Msg(text="x", buttons=[[Btn(text="b", callback_data="d")]], menu=MAIN_MENU)


def test_ranges_msg_captions_every_picture_with_its_own_point():
    """Подпись картинки принадлежит своей точке — порядок тот же, что у рисовальщика."""
    from harness.presentation import ranges_msg

    res = _mixed_result()
    msg = ranges_msg(["/data/ranges/1-1.png"], res)

    assert [photo.path for photo in msg.photos] == ["/data/ranges/1-1.png"]
    assert "колл шова" in msg.photos[0].caption


def test_range_photos_do_not_borrow_a_caption_from_another_point():
    """Лишний путь (разбор пересчитали) остаётся без подписи, а не с чужой."""
    from harness.presentation import range_photos

    res = _mixed_result()
    photos = range_photos(["/a.png", "/b.png"], res)

    assert photos[1].caption == ""


def test_ranges_msg_of_a_strict_hand_explains_why_there_is_no_picture():
    """Строгой точке рисовать нечего, и отказ называет причину, а не молчит."""
    from harness.presentation import ranges_msg

    strict = AnalysisResult(
        hand_no="H1",
        points=[_point(spot=SpotKind.PUSHFOLD_UNOPENED, ev_diff_bb=-1.0, zone=Zone.STRICT)],
        ranked=[0],
    )
    msg = ranges_msg([], strict)

    assert msg.photos == []
    assert "не опирается на догадку" in msg.text


def _alias(opponent_id: int, nick: str, links: int = 0):
    from harness.contracts import OpponentRecord

    return OpponentRecord(opponent_id=opponent_id, nick=nick, links=links)


def test_aliases_msg_of_the_longest_names_still_fits_one_telegram_message():
    """Список оппонентов обязан отправляться при любом их числе: `sendMessage`
    отказывает на 4096 символах, и не влезший список не пришёл бы игроку вовсе.
    """
    from harness.presentation import aliases_msg

    # 64 — предел ника, который игрок печатает руками (`bot.handlers`); схема
    # длину ника оппонента не ограничивает, но списку хватает и этой длины.
    long_list = [_alias(i, "я" * 64, links=99) for i in range(1, 201)]

    msg = aliases_msg(long_list)

    assert len(msg.text) <= 4096
    assert f"из {len(long_list)} — по алфавиту" in msg.text


def test_aliases_msg_counts_everyone_not_only_the_shown():
    """Знаменатель — сколько оппонентов у игрока, а не сколько поместилось."""
    from harness.presentation import aliases_msg

    aliases = [_alias(i, "я" * 64, links=1) for i in range(1, 101)]

    msg = aliases_msg(aliases)

    shown = sum(1 for line in msg.text.splitlines() if line.startswith("я"))
    assert shown < len(aliases)
    assert f"Показаны {shown} из {len(aliases)}" in msg.text


def test_aliases_msg_says_how_many_tournaments_each_opponent_is_bound_in():
    """Число турниров — единственный признак, что привязка состоялась: в файлах
    раздач участник обезличен, и ника за столом не видно.
    """
    from harness.presentation import aliases_msg

    assert "Vasya — турниров: 2" in aliases_msg([_alias(1, "Vasya", links=2)]).text


def test_aliases_msg_marks_the_numbers_that_stand_on_a_stitching():
    """Со второго турнира число держится словом владельца, а не данными.

    Один турнир — метка участника в нём и есть данные, сшивать нечего; два и
    больше — раздачи сложены утверждением «это один человек», и на экране это
    выглядит просто как выборка побольше.
    """
    from harness.presentation import aliases_msg

    one = aliases_msg([_alias(1, "Vasya", links=1)]).text
    two = aliases_msg([_alias(1, "Vasya", links=2)]).text

    assert "Vasya — турниров: 1" in one and "по вашей сшивке" not in one
    assert "Vasya — турниров: 2 · по вашей сшивке" in two


# --- Ответ на вопрос игрока ---------------------------------------------------------


def _question_results():
    """По одному результату на каждое имя словаря — вход `question_msg`."""
    from tests.test_question_text import ALL_RESULTS

    return ALL_RESULTS


def test_the_answer_names_the_calculation_it_came_from():
    """Подпись под ответом называет расчёт — у всех шести, а не у одного.

    Неверно выбранный расчёт выглядит нормальным ответом, просто не на тот
    вопрос: подпись — единственное, что делает подмену видимой игроку.
    """
    from harness.presentation import question_msg

    for result in _question_results():
        assert "Посчитано:" in question_msg(result, prose=None).text


def test_the_answer_carries_the_denominator_next_to_the_number():
    """Рядом с величиной — её выборка: «71.4% (5 из 7)», а не «71.4%»."""
    from harness.presentation import question_msg
    from tests.test_question_text import _frequency

    assert "71.4% (5 из 7)" in question_msg(_frequency(), prose=None).text


def test_the_answer_without_prose_still_shows_the_numbers():
    """Текст модели не прошёл проверку — числа расчёта всё равно у игрока.

    Прятать посчитанное кодом из-за фразы модели хуже, чем показать его без
    неё: то же решение, что у разбора без текста вердикта.
    """
    from harness.presentation import question_msg
    from tests.test_question_text import _frequency

    msg = question_msg(_frequency(), prose=None)

    assert "71.4%" in msg.text
    assert not msg.text.startswith("\n")
    assert "\n\n\n" not in msg.text


def test_the_prose_stands_above_the_numbers_it_came_from():
    from harness.presentation import question_msg
    from tests.test_question_text import _frequency

    msg = question_msg(_frequency(), prose="Вы ставите часто.")

    assert msg.text.index("Вы ставите часто.") < msg.text.index("Посчитано:")


def test_an_undecided_threshold_asks_for_the_missing_sample_not_a_verdict():
    """Интервал накрывает порог — наружу идёт недостающая выборка, а не вывод.

    Сам интервал при этом не печатается ни числом, ни словом (решение
    владельца): наружу идут величина, выборка, порог и то, чего не хватает.
    """
    from harness.contracts import ThresholdOutcome
    from harness.presentation import question_msg
    from tests.test_question_text import _threshold

    msg = question_msg(_threshold(ThresholdOutcome.UNDECIDED, 42), prose=None)

    assert "для вывода нужно около 42 наблюдений" in msg.text
    assert "выше порога" not in msg.text
    assert "ниже порога" not in msg.text


def test_a_measured_threshold_shows_its_own_sample():
    """Порог, взятый у поля, едет со своим знаменателем: он тоже измерен."""
    from harness.contracts import ThresholdOutcome
    from harness.presentation import question_msg
    from tests.test_question_text import _threshold

    msg = question_msg(_threshold(ThresholdOutcome.ABOVE, None), prose=None)

    assert "Порог — 61.0% (измерен по полю: 61 из 100)." in msg.text


def test_a_refusal_lists_what_can_be_counted_instead():
    """Отказ не пустой: он называет то, на что расчёт есть.

    И ни одной строки текста модели в нём нет — ответ без расчёта это общие
    знания о покере, а продукт продаёт посчитанное.
    """
    from harness.presentation import question_refusal_msg

    text = question_refusal_msg().text

    assert "нет расчёта" in text
    assert "Посчитать могу:" in text
    assert "против порога" in text


def test_a_question_answer_never_calls_a_decision_a_mistake():
    """Слово «ошиб» не печатается ни в одном ответе: расчёт судит против диапазона."""
    from harness.presentation import question_msg, question_refusal_msg

    texts = [question_msg(result, prose=None).text for result in _question_results()]
    texts.append(question_refusal_msg().text)

    assert not [text for text in texts if "ошиб" in text.lower()]


# --- сырые данные: что посчитал код, с подписью у каждого числа -----------------------


def test_the_raw_data_block_names_what_each_number_means():
    """Число без подписи — не диагностика, а шум: «9.1» не говорит ничего, «конечный
    банк 9.1 ББ» говорит всё. Тест держит ПОДПИСИ, а не числа: числа меняются от
    раздачи к раздачи, а обещание «здесь сказано, что это значит» — нет.
    """
    en = _postflop_hand()
    text = _deep_dive(analyze_hand(en), en).text
    for label in (
        "уровень",
        "блайнды",
        "анте",
        "Игроков в раздаче",
        "Вы:",
        "позиция",
        "стек до раздачи",
        "Борд",
        "Банк по улицам",
        "Конечный банк",
        "Исход раздачи для вас",
        "банк до хода",
        "доставить",
        "эфф. стек",
        "SPR",
        "живых",
        "вердикт",
        "зона",
        "Сверка денег с источником",
    ):
        assert label in text, f"число печатается без подписи: нет «{label}»"


def test_the_raw_data_block_prints_no_money_the_hand_does_not_contain():
    """Ни одной выдуманной суммы (CLAUDE.md). Проверяются именно ДЕНЬГИ — числа
    при «ББ»: номера точек, позиции и счётчики игроков деньгами не являются, и
    сверять их с суммами раздачи значило бы проверять не то правило.
    """
    en = _postflop_hand()
    hand = en.hand
    allowed = {round(v / hand.bb, 1) for v in en.report.pot_by_street.values()}
    allowed |= {round(en.report.final_pot / hand.bb, 1)}
    allowed |= {round(p.stack / hand.bb, 1) for p in hand.players}
    allowed |= {round(v / hand.bb, 1) for v in en.report.stacks_end.values()}
    allowed |= {round(pot.amount / hand.bb, 1) for pot in en.report.side_pots}
    allowed |= {abs(round(hero_stack_delta_bb(en), 1))}
    allowed |= {round(dp.to_call / hand.bb, 1) for dp in en.report.decision_points}
    allowed |= {round(dp.pot_before / hand.bb, 1) for dp in en.report.decision_points}
    allowed |= {round(dp.eff_stack / hand.bb, 1) for dp in en.report.decision_points}
    allowed |= {round(dp.eff_stack_bb, 1) for dp in en.report.decision_points}
    allowed |= {round(dp.action.committed_after / hand.bb, 1) for dp in en.report.decision_points}
    res = analyze_hand(en)
    allowed |= {abs(round(p.ev_diff_bb, 1)) for p in res.points}
    for point in res.points:
        if point.interval is not None:
            allowed |= {
                abs(round(point.interval.point_bb, 1)),
                abs(round(point.interval.low_bb, 1)),
                abs(round(point.interval.high_bb, 1)),
                abs(round(point.interval.cost_ceiling_bb, 1)),
            }
    allowed |= {abs(round(res.total_ev_loss_bb, 1))}

    text = _deep_dive(res, en).text
    money = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*ББ", text)]
    assert money, "в блоке не осталось ни одной суммы — тест перестал что-либо значить"
    assert set(money) <= allowed, f"выдуманные суммы: {set(money) - allowed}"


def test_every_detail_key_the_analysis_produces_has_a_label():
    """Ключ `detail` без подписи печатался бы машинным именем — и это не подпись.

    Гоняется настоящий `analyze_hand` по фикстурам всех расчётов: подпись
    обязана быть у каждого ключа, который ядро способно положить в точку.
    """
    from harness.presentation.messages import _DETAIL_LABELS
    from tests.test_preflop_analysis import (
        _make_facing_shove_hand,
        _make_hu_facing_shove_hand,
        _make_hu_shove_hand,
        _make_multiway_shove_hand,
    )
    from tests.test_river_analysis import _river_hand
    from tests.test_turn_flop_analysis import _hand as _turn_flop_hand

    hands = [
        _postflop_hand(),
        _make_hu_shove_hand(("Ah", "Ad"), 10.0),
        _make_hu_facing_shove_hand(10_000),
        _make_multiway_shove_hand(("Ah", "Ad"), 10.0, 3),
        _make_facing_shove_hand(("Ah", "Ad"), 10.0, 10.0),
        _river_hand(),
        _turn_flop_hand(),
    ]
    seen: set[str] = set()
    for en in hands:
        for point in analyze_hand(en).points:
            seen |= set(point.detail)
    assert seen, "ни один расчёт не положил ничего в detail — тест ничего не значит"
    # Ветки, до которых фикстуры не доходят: отказ солвера (`DidNotConverge`) и
    # лукап по чарту (`cheap_fold_verdict`). Прогоном их не собрать, поэтому они
    # перечислены здесь — второй половиной того же обещания.
    seen |= {"solver_error", "push_weight", "lookup_depth_bb"}
    # `unjudged` печатается своей строкой «вердикта нет: …», а не по таблице.
    printed_elsewhere = {RIVER_CALL_DETAIL, TURN_FLOP_CALL_DETAIL, "unjudged"}
    assert not (seen - printed_elsewhere) - set(_DETAIL_LABELS), (
        f"ключи без подписи: {sorted((seen - printed_elsewhere) - set(_DETAIL_LABELS))}"
    )


def test_a_detail_value_that_is_empty_prints_a_dash_not_a_blank():
    """Пустой список и отсутствующее значение — это «нечего показать», а не
    потерянное число; строка «подпись:» без единого знака после двоеточия
    читается вторым.
    """
    from harness.presentation.messages import _detail_value

    assert _detail_value([]) == "—"
    assert _detail_value({}) == "—"
    assert _detail_value(None) == "—"
    assert _detail_value(False) == "нет"
    assert _detail_value(0) == "0"
    assert _detail_value(0.0) == "0.00"
    assert _detail_value(-0.96) == "−0.96", "минус типографский, как у всех чисел продукта"
    assert _detail_value(0.000005) == "0.0000", "никакой экспоненциальной записи"


def test_no_engine_token_reaches_the_player_in_the_raw_block():
    """«fold»/«check» — язык движка, не игрока (SESSIONS_UX), и сырые числа от
    этого правила не освобождены: действие переводится и там."""
    en = _postflop_hand()
    text = _deep_dive(analyze_hand(en), en).text
    head = text.split("\nСверка денег", 1)[0]
    for token in ("сыграно: fold", "сыграно: check", "сыграно call", "сыграно check"):
        assert token not in head, f"токен движка дошёл до игрока: «{token}»"
    assert "сыграно: чек" in head


def test_the_hand_with_one_pot_says_nothing_about_side_pots():
    """`EngineReport.side_pots` держит ВСЕ поты PokerKit, включая главный, и на
    обычной раздаче там один элемент, равный конечному банку. Печатать его как
    «сайд-пот» значило бы утверждать деление, которого не было.
    """
    en = _postflop_hand()
    assert len(en.report.side_pots) == 1, "фикстура перестала быть однопотовой"
    assert "Банк делится на части" not in _deep_dive(analyze_hand(en), en).text


def test_a_hand_with_two_pots_names_each_part_and_who_claims_it():
    en = _two_side_pots_hand()
    assert len(en.report.side_pots) == 2, "фикстура перестала быть двухпотовой"
    line = next(
        line
        for line in _deep_dive(analyze_hand(en), en).text.splitlines()
        if line.startswith("Банк делится на части")
    )
    for pot in en.report.side_pots:
        assert f"{pot.amount / en.hand.bb:.1f} ББ (претендуют: {', '.join(pot.eligible)})" in line


def test_a_point_without_a_verdict_gets_no_zone_no_price_and_no_better_line():
    """`unjudged_point` конструирует `zone=strict, ev_diff_bb=0.0` как заглушки —
    печатать их значило бы выдать отсутствие расчёта за строгий нулевой вердикт.
    """
    unjudged = _point(
        spot=SpotKind.PREFLOP_OTHER, ev_diff_bb=0.0, zone=Zone.STRICT
    ).model_copy(update={"best_action": "", "detail": {"unjudged": "лимп модель не считает"}})
    text = _deep_dive(AnalysisResult(hand_no="TM1", points=[unjudged], ranked=[])).text
    assert "    вердикта нет: лимп модель не считает" in text
    assert "зона строго" not in text
    assert "цена" not in text
    assert "лучше" not in text
    assert "почему вердикта нет" not in text, "ключ напечатан своей строкой, не дважды"


def test_a_point_without_a_verdict_and_without_a_reason_still_says_so():
    silent = _point(
        spot=SpotKind.PREFLOP_OTHER, ev_diff_bb=0.0, zone=Zone.STRICT
    ).model_copy(update={"best_action": "", "detail": {}})
    text = _deep_dive(AnalysisResult(hand_no="TM1", points=[silent], ranked=[])).text
    assert "    вердикта нет." in text


def test_a_judged_point_keeps_its_zone_price_and_better_line():
    """Вторая сторона: у судимой точки всё это печатается."""
    text = _deep_dive(_mixed_result()).text
    assert "вердикт: спот пуш-фолд · зона строго · сыграно фолд · лучше шов · цена −2.3 ББ" in text


def test_the_price_of_a_point_never_renders_a_negative_zero():
    """`−0.03` после округления до десятой — честный ноль, а «−0.0» читается как
    отдельная (мнимая) отрицательная величина."""
    cheap = _point(
        spot=SpotKind.PUSHFOLD_UNOPENED, ev_diff_bb=-0.03, zone=Zone.STRICT
    )
    res = AnalysisResult(hand_no="TM1", points=[cheap], ranked=[0], total_ev_loss_bb=-0.03)
    text = _deep_dive(res).text
    assert "цена 0.0 ББ" in text
    assert "Сумма цены расхождений: 0.0 ББ" in text
    assert "−0.0" not in text


def test_the_action_of_a_point_prints_the_number_that_belongs_to_it():
    """`committed_after` — итог по улице, и у колла это не то число: доплата
    лежит в `to_call`. Чек и фолд суммы не несут вовсе."""
    en = _postflop_hand()
    text = _deep_dive(analyze_hand(en), en).text
    assert "сыграно: колл 2.0 ББ" in text, "колл печатает доплату"
    assert "сыграно: чек" in text and "сыграно: чек 0.0" not in text
    assert "сыграно: фолд" in text and "сыграно: фолд 0.0" not in text

    shove = _two_side_pots_hand()
    raise_text = _deep_dive(analyze_hand(shove), shove).text
    assert "сыграно: колл 4.4 ББ, олл-ин" in raise_text


def test_a_raise_prints_the_total_it_was_raised_to():
    from tests.test_preflop_analysis import _make_multiway_shove_hand

    en = _make_multiway_shove_hand(("7c", "2s"), 9.0, 3)
    assert "сыграно: рейз до 9.0 ББ, олл-ин" in _deep_dive(analyze_hand(en), en).text


def test_no_engine_token_of_the_analysis_reaches_the_player():
    """Значения `detail`, которые ядро пишет токенами движка, переводятся: иначе
    игрок читает `call_ev`, `unstable` и `fold` — язык расчёта, не его."""
    from harness.presentation.messages import _AXIS_WORD, _BRACKET_WORD, _METHOD_WORD

    assert _METHOD_WORD["call_ev"] == "EV колла против диапазона шовера"
    assert _METHOD_WORD["subset_enumeration"] == "перебор подмножеств ответивших"
    assert _METHOD_WORD["prefilter_chart_lookup"] == "лукап по чарту"
    assert _BRACKET_WORD == {"stable": "устойчива", "unstable": "через ноль"}
    assert set(_AXIS_WORD) == set(_BRACKET_WORD), "вторая ось говорит теми же значениями"

    from tests.test_preflop_analysis import _make_facing_shove_hand

    hands = [_make_facing_shove_hand(("Ah", "Ad"), 10.0, 10.0), _two_side_pots_hand()]
    texts = [_deep_dive(analyze_hand(en), en).text for en in hands]
    for text in texts:
        for token in ("call_ev", "subset_enumeration", ": stable", ": unstable", ": call", ": fold"):
            assert token not in text, f"токен движка дошёл до игрока: «{token}»"
    assert "×0.14" in texts[0], "ключ ширины печатается знаком умножения, не латинской x"
    assert "устойчивость к входу живых за вами: вердикт не меняется" in texts[1]


def test_every_validation_status_has_a_word_of_its_own():
    """Сверка денег печатается словом: `pass`/`reject` — это язык валидатора."""
    from harness.contracts import ValidationStatus
    from harness.presentation.messages import _VALIDATION_WORD

    assert set(_VALIDATION_WORD) == set(ValidationStatus)
    assert _VALIDATION_WORD[ValidationStatus.PASS] == "сошлась"
    assert _VALIDATION_WORD[ValidationStatus.ESCALATE] == "требует ответа игрока"
    assert _VALIDATION_WORD[ValidationStatus.REJECT] == "не сошлась"

    en = _postflop_hand()
    assert "Сверка денег с источником: сошлась." in _deep_dive(analyze_hand(en), en).text


def test_a_share_is_printed_as_a_percentage_not_as_a_fraction():
    """Доли и проценты в одном сообщении читатель принимает за разные величины."""
    from tests.test_preflop_analysis import _make_facing_shove_hand

    en = _make_facing_shove_hand(("Ah", "Ad"), 10.0, 10.0)
    text = _deep_dive(analyze_hand(en), en).text
    share_lines = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith(("требуемая эквити", "рук в диапазоне шова", "вероятность"))
    ]
    assert share_lines, "в фикстуре не оказалось ни одной доли — тест ничего не значит"
    for line in share_lines:
        assert "%" in line, f"доля напечатана дробью: {line}"


def test_the_ceiling_of_the_choice_is_named_only_where_the_interval_crosses_zero():
    """Потолок цены выбора отвечает на вопрос «сколько стоит ошибиться, когда оба
    варианта допустимы»; у интервала по одну сторону нуля этого вопроса нет."""
    across = _point(
        spot=SpotKind.PUSHFOLD_UNOPENED, ev_diff_bb=0.0, zone=Zone.STRICT
    ).model_copy(
        update={"interval": EvInterval(point_bb=-0.2, low_bb=-0.6, high_bb=0.4, near_zero=True)}
    )
    one_sided = _point(
        spot=SpotKind.PUSHFOLD_UNOPENED, ev_diff_bb=-1.0, zone=Zone.STRICT
    ).model_copy(
        update={"interval": EvInterval(point_bb=-1.0, low_bb=-1.4, high_bb=-0.6)}
    )
    text_across = _deep_dive(AnalysisResult(hand_no="A", points=[across], ranked=[0])).text
    text_one = _deep_dive(AnalysisResult(hand_no="B", points=[one_sided], ranked=[0])).text
    assert "потолок цены выбора 0.6 ББ" in text_across and "интервал через ноль" in text_across
    assert "потолок цены выбора" not in text_one


def test_a_stack_depth_is_printed_with_one_digit_like_every_other_stack():
    """Два знака у глубины и один у стека в соседней строке читатель принимает за
    разную точность измерения."""
    en = _two_side_pots_hand()
    text = _deep_dive(analyze_hand(en), en).text
    assert "глубина стека шовера, ББ: 25.0" in text
    assert "глубина стека шовера, ББ: 25.00" not in text


def test_the_total_price_stands_next_to_the_money_check_not_above_the_points():
    """Итог раздачи — итогом, а не шапкой: сумма и сверка денег стоят рядом, ниже
    всех точек, и обе несжимаемы."""
    lines = _deep_dive(_mixed_result()).text.splitlines()
    assert lines.index("Сумма цены расхождений: −3.4 ББ — только по оценённым точкам.") + 1 == (
        lines.index("Сверка денег с источником: сошлась.")
    )
    assert lines.index("Сверка денег с источником: сошлась.") > lines.index(
        next(line for line in lines if line.startswith("1. "))
    )


def test_the_scan_summary_counts_hands_grammatically():
    for total, word in ((1, "1 рука"), (4, "4 руки"), (7, "7 рук"), (11, "11 рук")):
        summary = ScanSummary(
            hands_total=total, hands_with_decision=0, items=[], total_loss_bb=0.0
        )
        assert scan_summary_msg(summary, quota_left=1, quota_total=5).text.startswith(
            f"Скан завершён: {word}."
        )


def test_the_pot_odds_of_the_raw_block_agree_with_the_calculation_tool():
    """`presentation` не имеет права импортировать `analysis` (образ бота тянул бы
    eval7 и pokerkit), поэтому формула шансов банка здесь своя — и сходиться с
    инструментом она обязана по тесту, а не по памяти правившего.
    """
    from harness.analysis.tools.pot_odds import required_equity
    from harness.presentation.messages import _required_equity

    pairs = [
        (100, 100), (1, 3), (169_000, 398_000), (69_000, 160_000), (7, 11),
        (300, 500), (50, 150), (2_500, 10_000), (1, 1), (999, 1_001),
    ]
    for to_call, pot_before in pairs:
        assert _required_equity(to_call, pot_before) == required_equity(to_call, pot_before)


def test_the_pot_odds_line_appears_only_where_there_is_something_to_call():
    """Шансы банка у бесплатного решения — не ноль, а отсутствие вопроса."""
    en = _postflop_hand()
    text = _deep_dive(analyze_hand(en), en).text
    lines = [line for line in text.splitlines() if "шансы банка" in line]
    free = [dp for dp in en.report.decision_points if dp.to_call == 0]
    priced = [dp for dp in en.report.decision_points if dp.to_call > 0]
    assert len(lines) == len(priced)
    assert free, "в фикстуре нет ни одной бесплатной точки — вторая половина не проверена"


def test_the_price_is_summed_only_where_something_was_judged():
    """Ноль несчитанной точки означает «не посчитано», а не «сыграно верно», и
    сумма, подписанная им, читалась бы как «потерь не было»."""
    unjudged = _point(
        spot=SpotKind.PREFLOP_OTHER, ev_diff_bb=0.0, zone=Zone.STRICT
    ).model_copy(update={"best_action": ""})
    silent = AnalysisResult(hand_no="TM1", points=[unjudged], ranked=[])
    loud = _mixed_result()
    assert "Сумма цены расхождений" not in _deep_dive(silent).text
    assert "Сумма цены расхождений" in _deep_dive(loud).text


def test_the_deep_dive_carries_no_words_of_the_model():
    """Решение владельца 2026-09-12: в разборе раздачи модель не участвует вовсе.

    Проверяется не отсутствие конкретной фразы, а отсутствие места, куда её
    можно было бы передать: у `hand_analysis_msgs` нет аргумента для текста модели.
    """
    from inspect import signature

    assert "verdict" not in signature(hand_analysis_msgs).parameters


# --- дверь в разбор раздачи: кнопка под КАЖДОЙ рукой скана ----------------------------


def _scan_of(hand_nos: list[str], items: list[ScanItem] | None = None) -> ScanSummary:
    return ScanSummary(
        hands_total=len(hand_nos),
        hands_with_decision=len(items or []),
        items=items or [],
        close_calls=[],
        total_loss_bb=0.0,
        hands_failed=0,
        points_total=len(hand_nos),
        points_judged=0,
        hand_nos=hand_nos,
    )


def test_a_hand_without_a_disagreement_still_has_a_way_into_its_analysis():
    """Кнопка «разобрать» — дверь в раздачу, а не награда за найденное расхождение.

    До этого теста кнопка ставилась ТОЛЬКО под строками списка расхождений. Файл,
    в котором расчёт не взялся судить ни одной точки (движок v1 судит только
    пуш-фолд), кнопок не получал вовсе — и разбор раздачи со всем, что в нём есть
    (блок «Что было», числа расчёта), был из HH-входа недостижим в принципе, на
    одной руке и на трёхстах одинаково.
    """
    msg = scan_summary_msg(_scan_of(["TM1"]), quota_left=1, quota_total=5)
    assert "deep:TM1" in [b.callback_data for row in msg.buttons for b in row]


def test_the_door_into_a_hand_is_offered_once_even_when_the_hand_is_in_the_list():
    """Рука с расхождением уже несёт свою кнопку — второй такой же быть не должно.

    Иначе под сводкой оказываются две одинаковые кнопки «разобрать» на одну
    раздачу, и игрок обязан гадать, чем они отличаются (ничем).
    """
    item = ScanItem(
        hand_no="TM1",
        hand_index=0,
        hero_class="AKo",
        spot=SpotKind.PUSHFOLD_UNOPENED,
        action_taken="fold",
        best_action="shove",
        ev_diff_bb=-1.2,
        zone=Zone.STRICT,
    )
    msg = scan_summary_msg(_scan_of(["TM1", "TM2"], items=[item]), quota_left=1, quota_total=5)
    data = [b.callback_data for row in msg.buttons for b in row]
    assert data.count("deep:TM1") == 1
    assert "deep:TM2" in data


def test_the_door_prefix_matches_the_one_the_router_listens_to():
    """`presentation` не импортирует роутер (правило зависимостей, CLAUDE.md), и
    две копии строки «deep:» держатся рядом только этим тестом.

    Разойдясь, они дали бы кнопку, на которую бот не отвечает ничем, кроме
    «Эта кнопка не работает» из `on_unhandled_callback`.
    """
    from harness.bot.router import DEEP_DIVE_PREFIX
    from harness.presentation.messages import _DEEP_PREFIX

    assert _DEEP_PREFIX == DEEP_DIVE_PREFIX
