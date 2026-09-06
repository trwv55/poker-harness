"""Текст отчёта по турниру — второй вход изложения в модель.

Владелец показал образец желаемого вывода, и он турнирного уровня, а не по одной
раздаче: сколько сыграно, как ходил стек, где ушли фишки, что расчёт судит и во
что это обошлось. Числа для всего этого уже посчитаны (`analysis/tournament.py`,
`TournamentReport`) — здесь только слова, и правила ровно те же, что у
`verdict_text`: выжимка чисел на вход, проверка чисел на выходе, отказ вместо
правки.

**Три вещи, ради которых выжимка устроена именно так.**

* *Разные величины не складываются.* `EvSplit` держит цену расхождений (EV на
  момент решения) отдельно от потерянных фишек (факт раздачи), и выжимка
  повторяет это разделение словами, а промпт запрещает их складывать. Иначе
  первая же связная фраза модели сложила бы их — они же «оба про потери».
* *Покрытие едет рядом с ценой.* Без него любая сумма читается как полная цена
  турнира, хотя судится обычно меньшая часть точек.
* *Скачок bb на границе уровня объясняется, а не прячется.* Стек на входе
  уровня меньше, чем на выходе предыдущего, при тех же фишках — блайнды
  выросли. Строка выглядит как ошибка в данных, поэтому такие переходы
  помечаются в выжимке отдельно и промпт требует объяснить их один раз словами
  (`test_a_blind_jump_between_levels_is_flagged_for_the_model`).
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from harness.contracts import (
    TournamentReport,
    TournamentTextOut,
    Zone,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import unsupported_numbers
from harness.explanation.verdict_text import Digest, UnfaithfulText, VerdictLLM

__all__ = ["tournament_digest", "tournament_text"]

_PROMPT_PATH = Path(__file__).parent / "prompts" / "tournament.md"

# Сколько строк каждого списка уходит в промпт. Потолок — не экономия ради
# экономии: отчёт по турниру на 150 раздач даёт десятки строк «где ушли фишки»,
# и модель, получив их все, пересказывает таблицу вместо того, чтобы назвать
# главное. Верхние строки уже отсортированы ядром по цене.
_MAX_ALL_INS = 5
_MAX_CHIP_MOVES = 5
_MAX_FINDINGS = 4
_MAX_LEVELS = 12


def _stats_lines(report: TournamentReport, book: NumberBook) -> list[str]:
    stats = report.stats
    lines = [
        (
            f"Раздач: {book.count(report.hands_total)}; "
            f"уровни с {book.count(report.first_level)} по {book.count(report.last_level)}; "
            f"время: {book.count(report.duration_minutes)} мин."
        )
    ]
    if report.hands_failed:
        lines.append(
            f"Не разобрано раздач: {book.count(report.hands_failed)} — в оценку не вошли."
        )
    shares = [
        ("добровольный вход в банк", stats.vpip_pct),
        ("повышение до флопа", stats.pfr_pct),
        ("ре-рейз до флопа", stats.reraise_pct),
        ("сдача на продолженную ставку", stats.fold_to_cbet_pct),
    ]
    lines.append(
        "Статистика турнира: "
        + "; ".join(
            f"{title} — {book.pct(value)}%" if value is not None else f"{title} — данных нет"
            for title, value in shares
        )
        + "."
    )
    baseline = report.baseline
    if baseline is None:
        lines.append("Среднего по прошлым турнирам нет: этот турнир в базе первый.")
    else:
        vpip = baseline.vpip_pct
        pfr = baseline.pfr_pct
        lines.append(
            f"В среднем по всем турнирам игрока (их {book.count(report.baseline_tournaments)}): "
            f"вход {book.pct(vpip) if vpip is not None else 'нет данных'}%, "
            f"повышение {book.pct(pfr) if pfr is not None else 'нет данных'}%."
        )
    return lines


def _trajectory_lines(report: TournamentReport, book: NumberBook) -> list[str]:
    trajectory = report.trajectory
    shown = trajectory.levels[-_MAX_LEVELS:]
    lines = ["Стек по уровням (в bb своего уровня):"]
    lines += [
        f"  ур. {book.count(level.level)}: "
        f"{book.bb(level.start_bb)} -> {book.bb(level.end_bb)} bb, "
        f"раздач {book.count(level.hands)}"
        for level in shown
    ]
    jumps = [
        book.count(nxt.level)
        for prev, nxt in pairwise(shown)
        if nxt.start_bb < prev.end_bb
    ]
    if jumps:
        lines.append(
            "  Уровни, где стек на входе меньше, чем на выходе предыдущего: "
            + ", ".join(jumps)
            + ". Это не ошибка данных: фишки те же, выросли блайнды."
        )
    lines.append(
        f"Максимум стека: {book.bb(trajectory.peak_bb)} bb на уровне "
        f"{book.count(trajectory.peak_level)} (раздача {book.token(trajectory.peak_hand_no)}); "
        f"после него сыграно раздач: {book.count(trajectory.hands_after_peak)}; "
        f"на выходе {book.bb(trajectory.final_bb)} bb, на входе в турнир "
        f"{book.bb(trajectory.start_bb)} bb."
    )
    return lines


def _all_in_lines(report: TournamentReport, book: NumberBook) -> list[str]:
    if not report.all_ins:
        return ["Олл-инов с участием героя не было."]
    lines = [f"Олл-ины (всего {book.count(len(report.all_ins))}, крупнейшие первыми):"]
    lines += [
        f"  раздача {book.token(event.hand_no)}, ур. {book.count(event.level)}"
        f"{f', рука {event.hero_class}' if event.hero_class else ''}: "
        f"вошёл с {book.bb(event.stack_before_bb)} bb, изменение "
        f"{book.bb(event.delta_bb)} bb"
        f"{', со вскрытием' if event.showdown else ''}"
        for event in report.all_ins[:_MAX_ALL_INS]
    ]
    return lines


def _chip_move_lines(report: TournamentReport, book: NumberBook) -> list[str]:
    if not report.chip_moves:
        return ["Раздач, в которых стек уменьшился, нет."]
    lines = ["Где ушли фишки (дороже первой):"]
    lines += [
        f"  раздача {book.token(move.hand_no)}, ур. {book.count(move.level)}"
        f"{f', рука {move.hero_class}' if move.hero_class else ''}: "
        f"{move.last_street.value}{', олл-ин' if move.all_in else ''}"
        f"{', вскрытие' if move.showdown else ''} — {book.bb(move.cost_bb)} bb"
        for move in report.chip_moves[:_MAX_CHIP_MOVES]
    ]
    return lines


def _finding_lines(report: TournamentReport, book: NumberBook) -> list[str]:
    if not report.findings:
        return ["Повторяющихся развилок среди оценённых решений не нашлось."]
    lines = ["Повторяющиеся развилки (только среди оценённых решений):"]
    for finding in report.findings[:_MAX_FINDINGS]:
        zone = "предполагая" if finding.zone is Zone.ASSUMING else "строго"
        lines.append(
            f"  спот {finding.spot.value}: сыграно «{finding.action_taken}», лучше "
            f"«{finding.best_action}»; повторов {book.count(finding.count)}, суммарно "
            f"{book.bb(finding.total_cost_bb)} bb; зона: {zone}."
        )
        if finding.seen_before:
            lines.append(
                f"    та же развилка в прошлых турнирах: точек "
                f"{book.count(finding.seen_before)} в "
                f"{book.count(finding.seen_before_tournaments)} турнирах."
            )
    return lines


def _ev_lines(report: TournamentReport, book: NumberBook) -> list[str]:
    ev = report.ev
    return [
        "Честный счёт (величины РАЗНЫЕ, складывать их нельзя):",
        f"  оценено решений: {book.count(ev.points_judged)} из {book.count(ev.points_total)};",
        (
            f"  цена расхождений в оценённых решениях (EV на момент решения): "
            f"{book.bb(ev.judged_loss_bb)} bb;"
        ),
        f"  фишек потеряно в раздачах с расхождением: {book.bb(ev.chips_in_gap_hands_bb)} bb;",
        (
            f"  фишек потеряно в проигранных олл-инах без расхождения (дисперсия, "
            f"решения расчёт не оспаривает): {book.bb(ev.chips_in_lost_allins_bb)} bb;"
        ),
        (
            f"  фишек потеряно в остальных раздачах (про них расчёт не говорит ничего): "
            f"{book.bb(ev.chips_elsewhere_bb)} bb."
        ),
    ]


def tournament_digest(report: TournamentReport) -> Digest:
    """Выжимка отчёта для промпта: факты, траектория, олл-ины, находки, счёт EV."""
    book = NumberBook()
    lines: list[str] = []
    for block in (
        _stats_lines(report, book),
        _trajectory_lines(report, book),
        _all_in_lines(report, book),
        _chip_move_lines(report, book),
        _finding_lines(report, book),
        _ev_lines(report, book),
    ):
        lines.extend(block)
        lines.append("")
    return Digest(text="\n".join(lines).strip(), allowed=book.allowed)


async def tournament_text(
    llm: VerdictLLM, report: TournamentReport, *, trace_id: int
) -> TournamentTextOut:
    """Рассказ по отчёту турнира. Один вызов модели, те же правила верности.

    Пустой ответ (модель не нашла что сказать) — это отказ, а не текст: пустые
    абзацы игроку не показываются, и молчаливое «ничего» неотличимо от поломки
    (`test_an_empty_answer_is_a_refusal_not_a_text`).
    """
    digest = tournament_digest(report)
    prompt = _PROMPT_PATH.read_text(encoding="utf-8").replace("{digest}", digest.text)
    draft, _meta = await llm("verdict_text", TournamentTextOut, prompt=prompt, trace_id=trace_id)

    paragraphs = [paragraph.strip() for paragraph in draft.paragraphs if paragraph.strip()]
    if not paragraphs:
        raise UnfaithfulText("модель вернула пустой отчёт по турниру")
    for index, paragraph in enumerate(paragraphs):
        invented = unsupported_numbers(paragraph, digest.allowed)
        if invented:
            raise UnfaithfulText(f"абзац {index + 1}: числа не из расчёта — {invented}")
    return TournamentTextOut(paragraphs=paragraphs)
