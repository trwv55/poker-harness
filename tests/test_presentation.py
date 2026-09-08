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

from harness.contracts import (
    AllInEvent,
    AnalysisResult,
    Assumption,
    ChipMove,
    EvInterval,
    EvSplit,
    Finding,
    LevelLine,
    PlayerStats,
    PointText,
    PointVerdict,
    Range,
    ScanItem,
    ScanSummary,
    SpotKind,
    StackTrajectory,
    Street,
    TournamentReport,
    TournamentTextOut,
    VerdictTextOut,
    Zone,
)
from harness.explanation import HandReplay, ReplaySpan
from harness.presentation import (
    Btn,
    Msg,
    bot_failure_msg,
    deep_dive_msg,
    escalation_msg,
    failed_msg,
    hh_accepted_msg,
    hh_duplicate_msg,
    new_session_msg,
    progress_text,
    quota_exceeded_msg,
    replay_msg,
    scan_summary_msg,
    start_msg,
    tournament_report_msg,
    tournament_story_msg,
    unsupported_document_msg,
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


def test_scan_summary_msg_always_reports_how_much_was_evaluated():
    """Покрытие печатается в обеих ветках — и со списком расхождений, и без него.

    Смесь обеих веток намеренная: шаблон, печатающий строку только в одной из
    них, тест не пройдёт (та же ловушка, что у пометки зоны и слова «ошибка»).
    """
    items = [_scan_item(hand_no="H1", ev_diff_bb=-2.3, zone=Zone.STRICT)]
    with_items = ScanSummary(
        hands_total=10,
        hands_with_decision=8,
        items=items,
        total_loss_bb=-2.3,
        points_total=40,
        points_judged=7,
    )
    without_items = ScanSummary(
        hands_total=10,
        hands_with_decision=8,
        items=[],
        total_loss_bb=0.0,
        points_total=40,
        points_judged=7,
    )

    assert "7 из 40" in scan_summary_msg(with_items, quota_left=1, quota_total=1).text
    assert "7 из 40" in scan_summary_msg(without_items, quota_left=1, quota_total=1).text


def test_scan_summary_msg_does_not_pass_an_empty_list_off_as_a_clean_game():
    """Пустой список — это «не нашли среди оценённых», а не «расхождений нет».

    После правил, снимающих вердикт с точки, которую нельзя посчитать честно,
    пустой список стал обычным исходом — и молчание о том, сколько решений
    осталось без оценки, читалось бы игроком как «сыграно чисто». Это ровно та
    деградация без огласки, которую сводка уже не допускает для `hands_failed`.
    """
    without_items = ScanSummary(
        hands_total=10,
        hands_with_decision=8,
        items=[],
        total_loss_bb=0.0,
        points_total=40,
        points_judged=7,
    )
    text = scan_summary_msg(without_items, quota_left=1, quota_total=1).text

    assert "оценённых" in text  # пустота — про оценённые решения, а не про турнир
    assert "33" in text  # сколько решений осталось без оценки — названо числом


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
    строится тем же f-строчным шаблоном и была сломана тем же образом (round 1),
    а затем несла то же нечестное «верно» в зоне `assuming` (round 2): у этой
    точки ядро знает только, что альтернатива выигрывала EV при угаданном
    диапазоне, а не что сыгранное было ошибкой.
    """
    res = _mixed_result()
    msg = deep_dive_msg(res, elapsed_s=1, zone=Zone.STRICT, quota_left=1, quota_total=1)

    assert "Префлоп · пуш-фолд: фолд (лучше: шов) — −2.3 bb" in msg.text
    assert (
        "Префлоп · колл шова: колл (лучше: шов) — −1.1 bb (по модели диапазонов)" in msg.text
    )
    assert "вместо шов" not in msg.text  # старая (грамматически сломанная) формулировка round 1
    assert "верно" not in msg.text  # старая (нечестная в assuming) формулировка round 2


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
        hh_duplicate_msg(),
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



def test_hh_duplicate_msg_offers_a_way_out():
    """Отказ от повторного разбора обязан назвать путь дальше — иначе игрок,
    которому ДЕЙСТВИТЕЛЬНО нужен тот же турнир заново, упирается в тупик."""
    assert "/new" in hh_duplicate_msg().text


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


def test_deep_dive_msg_renders_the_close_call_form_in_full():
    """Разбор точки «около нуля»: точка, интервал, вердикт и потолок цены.

    Это та самая форма, которой отказ «одного надёжного числа здесь нет» не
    давал: игрок видит и знак с порядком величины, и ширину интервала, и то,
    сколько максимум стоит выбор в любую сторону.
    """
    point = PointVerdict(
        dp_index=0,
        street=Street.PREFLOP,
        spot=SpotKind.PUSHFOLD_UNOPENED,
        zone=Zone.ASSUMING,
        action_taken="fold",
        best_action="около нуля, оба варианта допустимы",
        ev_diff_bb=0.0,
        interval=EvInterval(point_bb=0.1, low_bb=-0.3, high_bb=0.8, near_zero=True),
        assumption=Assumption(range=Range(weights={"AA": 1.0}), source="model:test"),
    )
    res = AnalysisResult(hand_no="H7", points=[point], ranked=[0], total_ev_loss_bb=0.0)
    text = deep_dive_msg(res, elapsed_s=3, zone=Zone.ASSUMING, quota_left=1, quota_total=1).text

    assert "шов или фолд" in text  # названы оба варианта, а не один «лучший»
    assert "около нуля, оба варианта допустимы" in text
    assert "0.1 bb" in text  # точечная оценка
    assert "−0.3 bb" in text and "0.8 bb" in text  # интервал
    assert "не больше 0.8 bb" in text  # потолок цены
    assert "(лучше:" not in text  # упрёка тут нет, и грамматика расхождения не применяется


def test_deep_dive_msg_names_the_call_side_for_a_close_call_facing_a_shove():
    """У колла чужого шова варианты другие — «колл или фолд», и модель другая."""
    point = PointVerdict(
        dp_index=0,
        street=Street.PREFLOP,
        spot=SpotKind.PUSHFOLD_FACING_SHOVE,
        zone=Zone.ASSUMING,
        action_taken="call",
        best_action="около нуля, оба варианта допустимы",
        ev_diff_bb=0.0,
        interval=EvInterval(point_bb=-0.2, low_bb=-1.4, high_bb=0.6, near_zero=True),
        assumption=Assumption(range=Range(weights={"AA": 1.0}), source="model:test"),
    )
    res = AnalysisResult(hand_no="H8", points=[point], ranked=[0], total_ev_loss_bb=0.0)
    text = deep_dive_msg(res, elapsed_s=3, zone=Zone.ASSUMING, quota_left=1, quota_total=1).text

    assert "колл или фолд" in text
    assert "не больше 1.4 bb" in text


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


# --- отчёт по турниру (задача 23) --------------------------------------------------


def _stats(**kw) -> PlayerStats:
    base = {
        "hands": 100,
        "vpip": 24,
        "pfr": 18,
        "reraise": 2,
        "reraise_chances": 24,
        "fold_to_cbet": 11,
        "cbet_faced": 20,
    }
    return PlayerStats(**{**base, **kw})


def _trajectory(levels: int = 2) -> StackTrajectory:
    return StackTrajectory(
        levels=[
            LevelLine(level=20 + i, hands=12, start_bb=34.0 - i, end_bb=33.0 - i)
            for i in range(levels)
        ],
        start_bb=34.0,
        final_bb=0.0,
        peak_level=23,
        peak_bb=41.2,
        peak_hand_no="TM777",
        hands_after_peak=50,
    )


def _finding(zone: Zone = Zone.STRICT, seen_before: int = 0, cost: float = 4.2) -> Finding:
    return Finding(
        spot=SpotKind.PUSHFOLD_FACING_SHOVE,
        action_taken="call",
        best_action="fold",
        zone=zone,
        count=3,
        total_cost_bb=cost,
        hand_nos=["H1", "H2", "H3"],
        seen_before=seen_before,
        seen_before_tournaments=2 if seen_before else 0,
    )


def _report(
    *,
    stats: PlayerStats | None = None,
    baseline: PlayerStats | None = None,
    baseline_tournaments: int = 1,
    trajectory: StackTrajectory | None = None,
    all_ins: list[AllInEvent] | None = None,
    chip_moves: list[ChipMove] | None = None,
    findings: list[Finding] | None = None,
    ev: EvSplit | None = None,
    hands_total: int = 146,
    hands_failed: int = 0,
) -> TournamentReport:
    return TournamentReport(
        hands_total=hands_total,
        hands_failed=hands_failed,
        levels_played=7,
        first_level=20,
        last_level=26,
        duration_minutes=72,
        stats=stats or _stats(),
        baseline=baseline,
        baseline_tournaments=baseline_tournaments,
        trajectory=trajectory or _trajectory(),
        all_ins=all_ins or [],
        chip_moves=chip_moves or [],
        findings=findings or [],
        ev=ev
        or EvSplit(
            judged_loss_bb=12.3,
            points_judged=67,
            points_total=402,
            chips_in_gap_hands_bb=45.0,
            chips_in_lost_allins_bb=68.0,
            chips_elsewhere_bb=22.5,
        ),
    )


def _all_in(hand_no: str = "TM1") -> AllInEvent:
    return AllInEvent(
        hand_no=hand_no,
        hand_index=3,
        level=23,
        hero_class="AKs",
        stack_before_bb=23.0,
        delta_bb=23.0,
        showdown=True,
    )


def _chip_move(hand_no: str = "TM1", cost_bb: float = 34.0) -> ChipMove:
    return ChipMove(
        hand_no=hand_no,
        hand_index=3,
        level=23,
        hero_class="K3s",
        last_street=Street.PREFLOP,
        all_in=True,
        showdown=True,
        cost_bb=cost_bb,
    )


def test_tournament_report_msg_states_hands_levels_and_duration():
    msg = tournament_report_msg(_report())
    assert "146" in msg.text
    assert "20" in msg.text and "26" in msg.text
    assert "1 ч 12 мин" in msg.text
    assert msg.buttons == []  # кнопки живут под сводкой скана, не здесь


def test_tournament_report_msg_says_plainly_there_is_nothing_to_compare_against():
    """Один турнир в базе — среднее совпало бы с самим турниром (бриф, дословно)."""
    msg = tournament_report_msg(_report(baseline=None, baseline_tournaments=1))
    assert "не с чем" in msg.text
    assert "24.0%" in msg.text  # сама статистика турнира при этом показана


def test_tournament_report_msg_shows_the_average_over_every_tournament():
    msg = tournament_report_msg(
        _report(
            baseline=_stats(hands=300, vpip=63, pfr=48),
            baseline_tournaments=3,
        )
    )
    assert "24.0%" in msg.text and "21.0%" in msg.text  # турнир и среднее рядом
    assert "не с чем" not in msg.text
    assert "3" in msg.text


def test_tournament_report_msg_does_not_print_a_missing_share_as_zero():
    """«0 из 0» — не ноль процентов, и печатать «0.0%» здесь значило бы соврать."""
    msg = tournament_report_msg(_report(stats=_stats(reraise=0, reraise_chances=0)))
    assert "Ре-рейз" in msg.text
    assert "0.0%" not in msg.text.split("Сдача")[0].split("Ре-рейз")[1]


def test_tournament_report_msg_names_the_turning_point_with_its_hand():
    msg = tournament_report_msg(_report())
    assert "41.2 bb" in msg.text
    assert "TM777" in msg.text
    assert "50" in msg.text  # раздач после максимума


def test_tournament_report_msg_always_prints_coverage():
    """Покрытие печатается всегда — и когда находки есть, и когда их нет."""
    with_findings = tournament_report_msg(_report(findings=[_finding()])).text
    without = tournament_report_msg(_report(findings=[])).text
    for text in (with_findings, without):
        assert "67" in text and "402" in text


def test_tournament_report_msg_never_calls_variance_a_mistake():
    """Дисперсия названа тем, что она есть: решение, которое расчёт не оспаривает.

    Слово «ошиб» не имеет права появиться ни в одной ветке (CLAUDE.md), а строка
    про проигранные олл-ины обязана прямо сказать, что расчёт эти решения не
    оспаривает — иначе число рядом с ценой расхождений читается как вторая цена
    ошибок.
    """
    with_findings = tournament_report_msg(
        _report(findings=[_finding()], all_ins=[_all_in()])
    ).text
    without = tournament_report_msg(_report()).text
    for text in (with_findings, without):
        assert "ошиб" not in text
        assert "не оспаривает" in text
    assert "расхожд" in with_findings


def test_tournament_report_msg_marks_assuming_findings_and_not_strict_ones():
    strict = tournament_report_msg(_report(findings=[_finding(zone=Zone.STRICT)])).text
    assuming = tournament_report_msg(_report(findings=[_finding(zone=Zone.ASSUMING)])).text
    assert "по модели диапазонов" not in strict
    assert "по модели диапазонов" in assuming


def test_tournament_report_msg_links_a_finding_to_past_tournaments():
    """«Тот самый паттерн, который мы разбирали» — со счётом, а не намёком.

    Голые цифры проверять нельзя: «5» и «2» встречаются в отчёте и сами по себе
    (покрытие, цены), и тест на них прошёл бы даже с вырезанной строкой —
    найдено фальсификацией.
    """
    msg = tournament_report_msg(_report(findings=[_finding(seen_before=5)]))
    assert "та же развилка в прошлых турнирах — точек: 5, турниров: 2" in msg.text
    without_history = tournament_report_msg(_report(findings=[_finding(seen_before=0)])).text
    assert "прошл" not in without_history


def test_tournament_report_msg_omits_a_hand_class_it_does_not_know():
    """Карт героя в источнике нет — сегмент исчезает, а не становится пустым."""
    move = _chip_move().model_copy(update={"hero_class": ""})
    text = tournament_report_msg(_report(chip_moves=[move], all_ins=[_all_in()])).text
    assert "№TM1 · ур. 23" in text
    assert " ·  · " not in text


def test_tournament_report_msg_says_when_a_list_is_trimmed():
    moves = [_chip_move(hand_no=f"H{i}", cost_bb=50.0 - i) for i in range(30)]
    msg = tournament_report_msg(_report(chip_moves=moves))
    shown = [line for line in msg.text.splitlines() if line.startswith("№H")]
    assert len(shown) < 30
    assert str(len(shown)) in msg.text and "30" in msg.text


def test_tournament_report_msg_of_a_long_tournament_fits_one_telegram_message():
    """Предел `sendMessage` — 4096 символов, и отказ на нём стоит игроку ретраев."""
    report = _report(
        trajectory=_trajectory(levels=30),
        all_ins=[_all_in(hand_no=f"TM{i}") for i in range(40)],
        chip_moves=[_chip_move(hand_no=f"H{i}", cost_bb=50.0 - i) for i in range(200)],
        findings=[_finding(seen_before=3, cost=9.0 - i) for i in range(12)],
    )
    assert len(tournament_report_msg(report).text) < 4096


def test_tournament_report_msg_surfaces_hands_it_could_not_analyse():
    """Пропуск виден строкой, а не только отсутствующей цифрой где-то в тексте."""
    assert "Раздач не разобрано: 3" in tournament_report_msg(_report(hands_failed=3)).text
    assert "не разобрано" not in tournament_report_msg(_report(hands_failed=0)).text


# --- задача 21: реплей, текст модели, рассказ по турниру -----------------------------


def _prose_result() -> AnalysisResult:
    """Две точки с РАЗНЫМИ `dp_index`: проза привязывается по индексу точки."""
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


def _verdict_text() -> VerdictTextOut:
    return VerdictTextOut(
        points=[
            PointText(dp_index=0, verdict_label="mistake", text="Шов здесь дороже фолда."),
            PointText(
                dp_index=3,
                verdict_label="mistake",
                text="Если оппонент шовит широко, колл дешевле.",
            ),
        ],
        summary="За раздачу расчёт нашёл два расхождения.",
    )


def _replay() -> HandReplay:
    return HandReplay(
        spans=[
            ReplaySpan(text="T1 · ур. 12 · 50/100\nHero SB J♥️9♥️ · 1 000 (10.0bb)\n\n"),
            ReplaySpan(text="ПРЕФЛОП · банк 210\nUTG фолд → "),
            ReplaySpan(text="Hero олл-ин 990 (9.9bb)", emphasis=True),
        ]
    )


def test_deep_dive_msg_puts_the_prose_under_the_point_it_explains():
    """Текст модели стоит под строкой СВОЕЙ точки — привязка по `dp_index`, а не
    по порядку: иначе пояснение уезжает под чужие числа."""
    msg = deep_dive_msg(
        _prose_result(), 12, Zone.ASSUMING, 17, 50, verdict=_verdict_text()
    )
    lines = msg.text.splitlines()
    first = next(i for i, line in enumerate(lines) if "пуш-фолд" in line)
    second = next(i for i, line in enumerate(lines) if "колл шова" in line)
    assert "Шов здесь дороже фолда." in lines[first + 1]
    assert "Если оппонент шовит широко" in lines[second + 1]


def test_deep_dive_msg_shows_the_summary_of_the_model():
    msg = deep_dive_msg(_prose_result(), 12, Zone.STRICT, 17, 50, verdict=_verdict_text())
    assert "За раздачу расчёт нашёл два расхождения." in msg.text


def test_deep_dive_msg_without_prose_has_no_holes_in_it():
    """Разбор без текста модели (её не позвали или текст не прошёл проверку) —
    цельное сообщение с числами, а не то же самое с пустыми местами."""
    msg = deep_dive_msg(_prose_result(), 12, Zone.STRICT, 17, 50)
    assert "\n\n\n" not in msg.text
    assert "−2.3 bb" in msg.text


def test_the_replay_left_the_verdict_message_for_the_details_button():
    """Ход раздачи ушёл под кнопку «Подробнее» (задача 23), а сама кнопка осталась.

    Реплей занимал в сообщении больше места, чем разбор, и упирался в предел
    `sendMessage` в 4096 символов вместе с прозой модели. Проверяются оба
    утверждения сразу: в вердикте хода нет, а нажать на него по-прежнему есть
    где — иначе «убрали» превратилось бы в «потеряли».
    """
    msg = deep_dive_msg(_prose_result(), 12, Zone.STRICT, 17, 50, verdict=_verdict_text())
    assert "ПРЕФЛОП" not in msg.text
    assert any(btn.callback_data.startswith("detail:") for row in msg.buttons for btn in row)


def test_replay_msg_marks_the_hero_decision_in_bold():
    """Точка решения героя выделена прямо в потоке действий (спека §5.6).

    Выделение — разметка Телеграма, поэтому у сообщения стоит `parse_mode`, и
    выделен ровно тот кусок, который пометил `explanation.hand_replay`.
    """
    msg = replay_msg(_replay(), "TM77")
    assert msg.parse_mode == "HTML"
    assert "<b>Hero олл-ин 990 (9.9bb)</b>" in msg.text
    assert "<b>ПРЕФЛОП" not in msg.text


def test_replay_msg_escapes_a_nickname_that_looks_like_a_tag():
    """Ники приходят со скрина: незакрытый `<` уронил бы отправку целиком."""
    replay = HandReplay(spans=[ReplaySpan(text="<script> & Hero"), ReplaySpan(text="шов", emphasis=True)])
    msg = replay_msg(replay, "TM<1>")
    assert "&lt;script&gt; &amp; Hero" in msg.text
    assert "<script>" not in msg.text


def test_deep_dive_msg_with_prose_fits_one_telegram_message():
    msg = deep_dive_msg(_prose_result(), 12, Zone.ASSUMING, 17, 50, verdict=_verdict_text())
    assert len(msg.text) < 4096


def test_tournament_report_msg_explains_the_bb_jump_between_levels():
    """Стек на входе уровня меньше, чем на выходе предыдущего, — строка выглядит
    ошибкой в счёте, и объяснение печатает КОД, а не модель."""
    jumped = StackTrajectory(
        levels=[
            LevelLine(level=20, hands=12, start_bb=34.0, end_bb=33.0),
            LevelLine(level=21, hands=12, start_bb=22.0, end_bb=21.0),
        ],
        start_bb=34.0,
        final_bb=21.0,
        peak_level=20,
        peak_bb=34.0,
        peak_hand_no="TM1",
        hands_after_peak=12,
    )
    assert "выросли блайнды" in tournament_report_msg(_report(trajectory=jumped)).text


def test_tournament_report_msg_stays_silent_about_a_jump_that_did_not_happen():
    assert "выросли блайнды" not in tournament_report_msg(_report()).text


def test_tournament_story_msg_keeps_paragraphs_apart():
    msg = tournament_story_msg(
        TournamentTextOut(paragraphs=["Первый абзац.", "Второй абзац."])
    )
    assert msg.text == "Первый абзац.\n\nВторой абзац."
    assert msg.buttons == []


def test_tournament_story_msg_says_when_it_had_to_cut():
    """Обрезка не бывает молчаливой — это то же правило, что у списков отчёта."""
    msg = tournament_story_msg(TournamentTextOut(paragraphs=["а" * 2000] * 4))
    assert len(msg.text) < 4096
    assert "из 4" in msg.text


def test_tournament_story_msg_counts_paragraphs_grammatically():
    """Числительное согласуется с существительным во всех трёх формах.

    Прежний шаблон писал «Показаны N абзаца» всегда и был верен ровно при N от 2
    до 4 — то есть тест на четырёх абзацах проходил случайно.
    """
    one = tournament_story_msg(TournamentTextOut(paragraphs=["а" * 3400, "б" * 2000]))
    assert "Показан 1 абзац из 2" in one.text

    # При пяти и больше подлежащее в родительном множественного, и сказуемое
    # встаёт в средний род единственного числа.
    five = tournament_story_msg(TournamentTextOut(paragraphs=["а" * 600] * 8))
    assert "Показаны" not in five.text
    assert "Показано 5 абзацев из 8" in five.text


# --- задача 22: тексты пути скриншота ----------------------------------------


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
    plain = deep_dive_msg(res, 12, Zone.STRICT, 17, 50)
    caveated = deep_dive_msg(
        res, 12, Zone.ASSUMING, 17, 50, not_checked=["сверка получателей банка"]
    )
    assert "Проверить на этом экране было нечем: сверка получателей банка." in caveated.text
    assert "Проверить на этом экране было нечем" not in plain.text
    assert "зона: строго" in plain.text
    assert "зона: строго" not in caveated.text


def test_a_decision_not_taken_is_explained_instead_of_the_generic_line():
    """Главный сценарий продукта не должен отвечать строкой ни о чём.

    Причина у ядра названа (`detail["unjudged_kind"]`), и до сообщения она не
    доходила: игрок, приславший стол в момент хода, получал «точек с вердиктом
    нет».
    """
    from harness.contracts import UNJUDGED_DECISION_NOT_TAKEN

    pending = _point(
        spot=SpotKind.PREFLOP_OTHER, ev_diff_bb=0.0, zone=Zone.STRICT
    ).model_copy(
        update={"best_action": "", "detail": {"unjudged_kind": UNJUDGED_DECISION_NOT_TAKEN}}
    )
    msg = deep_dive_msg(
        AnalysisResult(hand_no="TM1", points=[pending], ranked=[]), 5, None, 17, 50
    )
    assert "Решение по этой раздаче ещё не принято" in msg.text
    assert "точек с вердиктом нет" not in msg.text


def test_an_unjudged_point_without_a_known_reason_keeps_the_general_line():
    """Пересказать игроку внутреннюю формулировку ядра хуже, чем промолчать."""
    unpriced = _point(
        spot=SpotKind.PREFLOP_OTHER, ev_diff_bb=0.0, zone=Zone.STRICT
    ).model_copy(update={"best_action": "", "detail": {"unjudged": "перебор подмножеств"}})
    msg = deep_dive_msg(
        AnalysisResult(hand_no="TM1", points=[unpriced], ranked=[]), 5, None, 17, 50
    )
    assert "По этой раздаче точек с вердиктом нет." in msg.text
    assert "перебор подмножеств" not in msg.text


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
    assert "Суммарная потеря по всем точкам разбора: −6.3 bb." in msg.text
    assert "Сбрасывает против шова, где колл плюсовой — 3 раза, −4.1 bb" in msg.text


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


def _alias(alias_id: int, nick: str, links: int = 0):
    from harness.contracts import AliasRecord

    return AliasRecord(alias_id=alias_id, nick=nick, links=links)


def test_aliases_msg_of_the_longest_names_still_fits_one_telegram_message():
    """Список оппонентов обязан отправляться при любом их числе: `sendMessage`
    отказывает на 4096 символах, и не влезший список не пришёл бы игроку вовсе.
    """
    from harness.presentation import aliases_msg

    # 64 — предел колонки `player_aliases.opponent_nick`, то же число, что у
    # собственного ника игрока (`players.gg_nickname`).
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
