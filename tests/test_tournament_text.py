"""Текст отчёта по турниру (задача 21, второй вход в модель) — без сети.

Проверяется код вокруг модели: что уехало в выжимку и что принимается обратно.
Отчёт (`TournamentReport`) здесь собирается прямо — он ВХОД, и считать его
настоящим `tournament_report` заново незачем: тот проверен в
`test_tournament_report.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar, cast

import pytest
from pydantic import BaseModel

from harness.contracts import (
    AllInEvent,
    ChipMove,
    EvSplit,
    Finding,
    LevelLine,
    PlayerStats,
    SpotKind,
    StackTrajectory,
    Street,
    TournamentReport,
    TournamentTextOut,
    Zone,
)
from harness.explanation.faithfulness import numbers_in
from harness.explanation.tournament_text import tournament_digest, tournament_text
from harness.explanation.verdict_text import UnfaithfulText

_T = TypeVar("_T", bound=BaseModel)


class FakeLLM:
    def __init__(self, reply: TournamentTextOut) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def __call__(
        self,
        purpose: Literal["verdict_text"],
        schema: type[_T],
        *,
        prompt: str,
        images: Sequence[bytes] = (),
        trace_id: int,
    ) -> tuple[_T, None]:
        self.prompts.append(prompt)
        return cast("_T", self.reply), None


def _report(
    *,
    levels: list[LevelLine] | None = None,
    findings: list[Finding] | None = None,
    baseline: PlayerStats | None = None,
) -> TournamentReport:
    levels = levels or [
        LevelLine(level=10, hands=12, start_bb=25.0, end_bb=31.0),
        LevelLine(level=11, hands=9, start_bb=20.5, end_bb=8.2),
    ]
    return TournamentReport(
        hands_total=21,
        hands_failed=0,
        levels_played=len(levels),
        first_level=levels[0].level,
        last_level=levels[-1].level,
        duration_minutes=47,
        stats=PlayerStats(hands=21, vpip=5, pfr=4, reraise=1, reraise_chances=3),
        baseline=baseline,
        baseline_tournaments=2 if baseline is not None else 1,
        trajectory=StackTrajectory(
            levels=levels,
            start_bb=25.0,
            final_bb=8.2,
            peak_level=10,
            peak_bb=31.0,
            peak_hand_no="TM7",
            hands_after_peak=9,
        ),
        all_ins=[
            AllInEvent(
                hand_no="TM19",
                hand_index=19,
                level=11,
                # Класс руки с цифрой — намеренно: «A9s» несёт 9, и прежняя
                # фикстура «KK» не могла поймать незарегистрированное число
                # (ревью, раздел A).
                hero_class="A9s",
                stack_before_bb=18.4,
                delta_bb=-18.4,
                showdown=True,
            )
        ],
        chip_moves=[
            ChipMove(
                hand_no="TM19",
                hand_index=19,
                level=11,
                hero_class="A9s",
                last_street=Street.PREFLOP,
                all_in=True,
                showdown=True,
                cost_bb=18.4,
            )
        ],
        findings=findings
        if findings is not None
        else [
            Finding(
                spot=SpotKind.PUSHFOLD_UNOPENED,
                action_taken="fold",
                best_action="shove",
                zone=Zone.ASSUMING,
                count=3,
                total_cost_bb=2.7,
                hand_nos=["TM3", "TM8", "TM14"],
                seen_before=2,
                seen_before_tournaments=1,
            )
        ],
        ev=EvSplit(
            judged_loss_bb=2.9,
            points_judged=11,
            points_total=17,
            chips_in_gap_hands_bb=4.1,
            chips_in_lost_allins_bb=18.4,
            chips_elsewhere_bb=3.3,
        ),
    )


def _answer(*paragraphs: str) -> TournamentTextOut:
    return TournamentTextOut(paragraphs=list(paragraphs))


# --- выжимка -------------------------------------------------------------------------


def test_the_digest_registers_every_number_it_prints():
    digest = tournament_digest(_report(baseline=PlayerStats(hands=60, vpip=14, pfr=9)))
    unregistered = [n for n in numbers_in(digest.text) if round(n, 1) not in digest.allowed]
    assert unregistered == []


def test_the_three_chip_buckets_and_the_ev_price_stay_separate():
    """Четыре величины `EvSplit` названы порознь и подписаны — иначе связный
    рассказ сложит цену расхождений с проигранными фишками."""
    text = tournament_digest(_report()).text
    assert "цена расхождений" in text and "дисперсия" in text
    assert "складывать их нельзя" in text


def test_coverage_travels_next_to_the_price():
    """Покрытие названо ТОЧКАМИ РЕШЕНИЯ, а не раздачами: первый живой прогон дал
    «расчёт оценил 36 решений из 185 сыгранных раздач» — модель прочла число
    точек как число раздач, потому что подпись это допускала."""
    text = tournament_digest(_report()).text
    assert "точек решения героя за турнир: 17, из них с вердиктом: 11" in text
    assert "не раздачи" in text


def test_no_engine_token_reaches_the_model():
    """Английские токены движка в промпт не идут вовсе. Первый живой прогон:
    модель переписала `pushfold_unopened` и `preflop` в текст игроку как есть —
    надёжнее всего этого не случается, когда она их не видела."""
    text = tournament_digest(_report()).text
    for token in ("pushfold_unopened", "preflop", "fold", "shove", "call"):
        assert token not in text


def test_a_blind_jump_between_levels_is_flagged_for_the_model():
    """Стек на входе уровня меньше, чем на выходе предыдущего, — строка выглядит
    как ошибка данных, и модель обязана получить объяснение, а не догадку."""
    text = tournament_digest(_report()).text
    assert "выросли блайнды" in text
    assert "меньше, чем на выходе предыдущего: 11" in text


def test_levels_without_a_jump_are_not_flagged():
    """Пометки нет там, где скачка нет: иначе объяснение появлялось бы всегда и
    ничего не значило."""
    text = tournament_digest(
        _report(
            levels=[
                LevelLine(level=10, hands=12, start_bb=25.0, end_bb=31.0),
                LevelLine(level=11, hands=9, start_bb=31.0, end_bb=8.2),
            ]
        )
    ).text
    assert "выросли блайнды" not in text


def test_the_first_tournament_says_there_is_nothing_to_compare_with():
    assert "турнир в базе первый" in tournament_digest(_report()).text


# --- вызов модели --------------------------------------------------------------------


async def test_the_prompt_carries_the_digest_and_the_rules():
    llm = FakeLLM(_answer("Турнир прошёл ровно."))
    await tournament_text(llm, _report(), trace_id=1)
    assert "Не выдумывай числа" in llm.prompts[0]
    assert "Стек по уровням" in llm.prompts[0]
    assert "{digest}" not in llm.prompts[0]


async def test_numbers_from_the_digest_pass():
    llm = FakeLLM(_answer("Стек дошёл до 31.0 bb, а вышли вы с 8.2 bb."))
    out = await tournament_text(llm, _report(), trace_id=1)
    assert len(out.paragraphs) == 1


async def test_an_invented_number_rejects_the_report_text():
    llm = FakeLLM(_answer("Вы потеряли 44.4 bb за турнир."))
    with pytest.raises(UnfaithfulText, match="44.4"):
        await tournament_text(llm, _report(), trace_id=1)


async def test_an_empty_answer_is_a_refusal_not_a_text():
    llm = FakeLLM(_answer("", "   "))
    with pytest.raises(UnfaithfulText):
        await tournament_text(llm, _report(), trace_id=1)
