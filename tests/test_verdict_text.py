"""Текст вердикта (задача 21): выжимка, структурная метка, отказ от выдумки.

**В сеть тесты не ходят** — то же ограничение, что у задачи 16: модель
подменяется двойником `FakeLLM`, который возвращает заранее заданный ответ и
запоминает промпт. Здесь проверяется КОД вокруг модели (что уходит в промпт,
что принимается обратно), а не качество текста: качество — этаж 3, eval-прогон
`evals/verdict/`, и он в набор тестов не входит.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar, cast

import pytest
from pydantic import BaseModel

from harness.contracts import (
    AnalysisResult,
    Assumption,
    EvInterval,
    PointVerdict,
    Range,
    SpotKind,
    Street,
    Zone,
)
from harness.explanation.faithfulness import numbers_in
from harness.explanation.verdict_text import (
    UnfaithfulText,
    VerdictDraft,
    verdict_digest,
    verdict_text,
)

_T = TypeVar("_T", bound=BaseModel)


class FakeLLM:
    """Двойник фасада: отдаёт заданный черновик и запоминает, что его просили.

    Реализует ровно протокол `VerdictLLM` — если протокол разойдётся с фасадом,
    это увидит pyright на настоящем вызывающем (`worker.pipeline`), а не тест.
    """

    def __init__(self, reply: VerdictDraft) -> None:
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
        assert purpose == "verdict_text"
        self.prompts.append(prompt)
        return cast("_T", self.reply), None


def _point(
    *,
    dp_index: int,
    ev_diff_bb: float,
    zone: Zone = Zone.STRICT,
    spot: SpotKind = SpotKind.PUSHFOLD_UNOPENED,
    best_action: str = "shove",
    interval: EvInterval | None = None,
    detail: dict[str, object] | None = None,
) -> PointVerdict:
    return PointVerdict(
        dp_index=dp_index,
        street=Street.PREFLOP,
        spot=spot,
        zone=zone,
        action_taken="fold",
        best_action=best_action,
        ev_diff_bb=ev_diff_bb,
        interval=interval,
        assumption=(
            Assumption(
                range=Range(weights={"AA": 1.0, "KK": 0.5}),
                source="model:multiway_pushfold",
                # Цифра в пояснении — намеренно: `preflop` пишет туда и глубину,
                # и число живых позади, а первый живой прогон отбраковал текст
                # ровно за такое число, показанное модели и не зарегистрированное.
                note="колл-диапазон на 12bb, 2 живых позади",
            )
            if zone is Zone.ASSUMING
            else None
        ),
        detail=detail or {},
    )


def _result(points: list[PointVerdict], ranked: list[int] | None = None) -> AnalysisResult:
    return AnalysisResult(
        hand_no="TM99",
        points=points,
        ranked=list(range(len(points))) if ranked is None else ranked,
        total_ev_loss_bb=sum(p.ev_diff_bb for p in points if p.ev_diff_bb < 0),
    )


def _draft(*texts: tuple[int, str], summary: str = "Итог без чисел.") -> VerdictDraft:
    return VerdictDraft.model_validate(
        {"points": [{"dp_index": i, "text": t} for i, t in texts], "summary": summary}
    )


# --- выжимка -------------------------------------------------------------------------


def test_the_digest_registers_every_number_it_prints():
    """Каждое число промпта разрешено в ответе — иначе модель наказана за то, что
    процитировала нас же. Инвариант механический: числа промпта минус
    разрешённые обязаны дать пустоту."""
    res = _result(
        [
            _point(
                dp_index=0,
                ev_diff_bb=-1.23,
                zone=Zone.ASSUMING,
                interval=EvInterval(point_bb=-1.23, low_bb=-3.0, high_bb=0.4),
                detail={"shover_depth_bb": 12.25, "live_others": 2},
            ),
            _point(dp_index=1, ev_diff_bb=-0.4, detail={"lookup_depth_bb": 9.5}),
        ]
    )
    digest = verdict_digest(res)
    # Заголовок точки («Точка 1 (dp_index 3)») несёт ИДЕНТИФИКАТОРЫ, а не
    # величины, и в реестр не идёт сознательно — см. `_point_lines`.
    body = [line for line in digest.text.splitlines() if not line.startswith("Точка ")]
    unregistered = [
        n for n in numbers_in("\n".join(body)) if round(n, 1) not in digest.allowed
    ]
    assert unregistered == []


def test_a_point_without_a_verdict_never_reaches_the_model():
    """В выжимку идут только точки из `ranked` — то есть только судимые."""
    judged = _point(dp_index=0, ev_diff_bb=-1.2)
    unjudged = _point(dp_index=7, ev_diff_bb=0.0, spot=SpotKind.POSTFLOP, best_action="")
    digest = verdict_digest(_result([judged, unjudged], ranked=[0]))
    assert "dp_index 0" in digest.text
    assert "dp_index 7" not in digest.text


def test_no_engine_token_reaches_the_model():
    """То же, что в выжимке турнира: `fold`/`shove` переводятся до промпта, а
    источник допущения (`model:multiway_pushfold`) в промпт не идёт вовсе —
    в нём английский токен, который модель раньше переписывала игроку как есть."""
    digest = verdict_digest(
        _result(
            [
                _point(dp_index=0, ev_diff_bb=-1.2),
                _point(dp_index=1, ev_diff_bb=-0.9, zone=Zone.ASSUMING),
            ]
        )
    )
    assert "fold" not in digest.text and "shove" not in digest.text
    assert "model:" not in digest.text and "multiway" not in digest.text
    assert "фолд" in digest.text and "шов" in digest.text


def test_the_digest_carries_the_interval_and_the_ceiling():
    """Интервал и потолок цены обязаны доехать до модели: без них она не сможет
    сказать про устойчивость вывода ничего, кроме выдуманного."""
    digest = verdict_digest(
        _result(
            [
                _point(
                    dp_index=0,
                    ev_diff_bb=0.0,
                    interval=EvInterval(point_bb=0.0, low_bb=-0.3, high_bb=0.8, near_zero=True),
                )
            ]
        )
    )
    assert "-0.3" in digest.text and "0.8" in digest.text
    assert "около нуля: да" in digest.text


def test_only_the_whitelisted_detail_keys_reach_the_model():
    """Из `detail` в промпт идут три величины и ни одной больше: остальное —
    внутренняя кухня расчёта, и показать её значит разрешить её назвать."""
    digest = verdict_digest(
        _result(
            [
                _point(
                    dp_index=0,
                    ev_diff_bb=-1.2,
                    detail={
                        "shover_depth_bb": 12.25,
                        "lookup_depth_bb": 9.5,
                        "live_others": 2,
                        "equilibrium_hand_regret_bb": 0.004,
                        "shove_range_fraction": 0.317,
                        "method": "call_ev",
                    },
                )
            ]
        )
    )
    assert "12.2" in digest.text and "9.5" in digest.text and "позади: 2" in digest.text
    assert "0.004" not in digest.text and "0.317" not in digest.text
    assert "call_ev" not in digest.text


async def test_a_small_round_number_is_not_allowed_by_the_point_numbering():
    """«около 1 bb» на точке ценой −1.4 bb — выдумка, и нумерация точек не имеет
    права её оправдывать (ревью, раздел A: `book.count(ordinal)` разрешал 0 и 1
    в любом разборе)."""
    res = _result([_point(dp_index=0, ev_diff_bb=-1.4), _point(dp_index=1, ev_diff_bb=-2.3)])
    llm = FakeLLM(_draft((0, "Фолд стоил около 1 bb."), (1, "Дорогое расхождение.")))
    with pytest.raises(UnfaithfulText, match="1.0"):
        await verdict_text(llm, res, trace_id=1)


# --- вызов модели --------------------------------------------------------------------


async def test_no_judged_points_means_no_model_call():
    """Судить нечего — модель не зовётся вовсе: ни токенов, ни риска выдумки."""
    llm = FakeLLM(_draft())
    out = await verdict_text(llm, _result([], ranked=[]), trace_id=1)
    assert llm.prompts == []
    assert out.points == [] and out.summary == ""


async def test_the_label_comes_from_the_core_not_from_the_prose():
    """Метка ставится по цене ядра, а не по тону текста: хвалебный текст на точке
    ценой −1.2 bb всё равно получает метку расхождения."""
    res = _result([_point(dp_index=0, ev_diff_bb=-1.2), _point(dp_index=1, ev_diff_bb=0.0)])
    llm = FakeLLM(
        _draft((0, "Отличный фолд, всё сделано правильно."), (1, "Здесь всё в порядке."))
    )
    out = await verdict_text(llm, res, trace_id=1)
    assert [p.verdict_label for p in out.points] == ["mistake", "ok"]


async def test_the_prompt_contains_the_digest_and_the_rules():
    res = _result([_point(dp_index=0, ev_diff_bb=-1.2)])
    llm = FakeLLM(_draft((0, "Шов здесь дороже на 1.2 bb.")))
    await verdict_text(llm, res, trace_id=1)
    prompt = llm.prompts[0]
    assert "Раздача TM99" in prompt
    assert "Не выдумывай числа" in prompt
    assert "{digest}" not in prompt


async def test_points_are_returned_in_the_ranked_order():
    """Порядок — из `ranked` (самая дорогая первой), а не тот, в каком ответила
    модель: порядок показа принадлежит расчёту."""
    res = _result(
        [_point(dp_index=0, ev_diff_bb=-0.3), _point(dp_index=1, ev_diff_bb=-2.0)],
        ranked=[1, 0],
    )
    llm = FakeLLM(_draft((0, "Мелкое расхождение."), (1, "Дорогое расхождение.")))
    out = await verdict_text(llm, res, trace_id=1)
    assert [p.dp_index for p in out.points] == [1, 0]


# --- отказ от неверного текста --------------------------------------------------------


async def test_an_invented_number_rejects_the_whole_text():
    """Число, которого нет в расчёте, — отказ: игрок получит разбор без прозы,
    но не получит выдуманную цифру про свои деньги."""
    res = _result([_point(dp_index=0, ev_diff_bb=-1.2)])
    llm = FakeLLM(_draft((0, "Шов дороже фолда примерно на 3.7 bb.")))
    with pytest.raises(UnfaithfulText, match="3.7"):
        await verdict_text(llm, res, trace_id=1)


async def test_an_invented_number_in_the_summary_is_caught_too():
    res = _result([_point(dp_index=0, ev_diff_bb=-1.2)])
    llm = FakeLLM(_draft((0, "Шов дороже."), summary="Всего за раздачу ушло 9.9 bb."))
    with pytest.raises(UnfaithfulText, match="9.9"):
        await verdict_text(llm, res, trace_id=1)


async def test_a_number_taken_from_the_digest_passes():
    res = _result(
        [
            _point(
                dp_index=0,
                ev_diff_bb=-1.2,
                interval=EvInterval(point_bb=-1.2, low_bb=-2.4, high_bb=0.5),
            )
        ]
    )
    llm = FakeLLM(_draft((0, "Фолд стоил 1.2 bb, в худшем случае 2.4 bb.")))
    out = await verdict_text(llm, res, trace_id=1)
    assert out.points[0].text.startswith("Фолд стоил")


async def test_a_reproach_on_a_near_zero_point_is_rejected():
    """«Лучше было» на точке, где расчёт не спорит ни с одним из вариантов, —
    утверждение, которого расчёт не делал. Это верность, а не тон, поэтому отказ."""
    res = _result(
        [
            _point(
                dp_index=0,
                ev_diff_bb=0.0,
                interval=EvInterval(point_bb=0.1, low_bb=-0.3, high_bb=0.4, near_zero=True),
            )
        ]
    )
    llm = FakeLLM(_draft((0, "Здесь лучше было пасовать.")))
    with pytest.raises(UnfaithfulText, match="упрекает"):
        await verdict_text(llm, res, trace_id=1)


async def test_a_near_zero_point_without_a_reproach_passes():
    res = _result(
        [
            _point(
                dp_index=0,
                ev_diff_bb=0.0,
                interval=EvInterval(point_bb=0.1, low_bb=-0.3, high_bb=0.4, near_zero=True),
            )
        ]
    )
    llm = FakeLLM(_draft((0, "Оба варианта допустимы, выбор дёшев.")))
    out = await verdict_text(llm, res, trace_id=1)
    assert out.points[0].verdict_label == "ok"


async def test_an_assuming_point_without_assumption_words_is_rejected():
    """Зона «предполагая» обязана быть названа словами — иначе догадка подаётся
    как факт, а это ровно то, против чего стоит вся система зон."""
    res = _result([_point(dp_index=0, ev_diff_bb=-1.2, zone=Zone.ASSUMING)])
    llm = FakeLLM(_draft((0, "Шов здесь дороже фолда на 1.2 bb.")))
    with pytest.raises(UnfaithfulText, match="допущение"):
        await verdict_text(llm, res, trace_id=1)


async def test_a_strict_point_needs_no_assumption_words():
    res = _result([_point(dp_index=0, ev_diff_bb=-1.2, zone=Zone.STRICT)])
    llm = FakeLLM(_draft((0, "Шов здесь дороже фолда на 1.2 bb.")))
    out = await verdict_text(llm, res, trace_id=1)
    assert out.points[0].verdict_label == "mistake"


async def test_an_answer_about_the_wrong_points_is_rejected():
    """Модель ответила не про те точки — сопоставлять по порядку нельзя: текст
    уехал бы под чужие числа."""
    res = _result([_point(dp_index=0, ev_diff_bb=-1.2)])
    llm = FakeLLM(_draft((5, "Про какую-то другую точку.")))
    with pytest.raises(UnfaithfulText, match="точкам"):
        await verdict_text(llm, res, trace_id=1)
