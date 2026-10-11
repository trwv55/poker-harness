"""Маршрутизация вопроса игрока к расчёту: модель выбирает имя, считает код.

Шесть расчётов словаря подключены модели инструментами. Модель не пишет SQL и
не считает: она называет инструмент и подставляет параметры, инструмент зовёт
`calcs.run` и возвращает выжимку его выхода, модель излагает эту выжимку
словами. Всё, что в ответе окажется сверх выхода инструмента, отбраковывается
(`explanation.question.check_answer`).

**Четыре ограничения, и каждое здесь механическое, а не в тексте промпта.**

1. *Число только из выхода инструмента.* Разрешённые числа — выход того
   вызова, который действительно состоялся (`CalcRun.digest`), а не всё, что
   модель могла бы вспомнить; непрошедший текст теряется целиком
   (`test_a_number_the_tool_did_not_return_costs_the_whole_prose`).
2. *Ответ называет расчёт.* `CalcRun.result` возвращается вызывающему, и подпись
   собирает `presentation` по полю `CalcResult.calc` — модель в подписи не
   участвует (`test_the_answer_names_the_calculation_it_came_from`).
3. *Один вопрос — один расчёт.* Второй успешный вызов инструмента не
   выполняется вовсе: он возвращает отказ, а первый результат остаётся
   единственным (`test_the_second_calculation_is_refused`). Перебор срезов
   запрещён не уговором: нарежь достаточно мелко, и отклонение найдётся по
   случайности. Это правило `/ask`; у агента потолок 8
   (`CalcRun(max_calcs=…)`, `test_the_ninth_calculation_is_refused`).
4. *Нет подходящего расчёта — отказ.* Ответ без единого вызова инструмента
   игроку не показывается вовсе (`result is None`,
   `test_a_question_without_a_calculation_is_refused`), сколько бы уверенно он
   ни звучал: сказать что-то о покере, ничего не посчитав, модель может всегда.

**Почему пакет `calcs`, а не `explanation`.** Инструменту нужна база, а
конвейерные пакеты про базу не знают (CLAUDE.md, правило зависимостей).
Выжимка и проверка ответа живут в `explanation.question` — они чистые, — а
склейка «база + модель» здесь, рядом со словарём, который она вызывает.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel
from pydantic_ai import ToolReturn
from pydantic_ai.tools import Tool
from sqlalchemy.ext.asyncio import AsyncSession

from harness.calcs import run
from harness.contracts import (
    BetSizeThreshold,
    CalcName,
    CalcParams,
    CalcResult,
    CoverageParams,
    DefenseParams,
    FieldThreshold,
    FrequencyStat,
    HeroFrequencyParams,
    LeaksParams,
    OpponentFrequencyParams,
    PointFilter,
    SpotKind,
    Street,
    Subject,
    ThresholdParams,
    ThresholdSide,
    Window,
    Zone,
)
from harness.explanation.question import (
    QuestionDraft,
    check_answer,
    question_digest,
    question_prompt,
)
from harness.explanation.verdict_text import Digest, UnfaithfulText
from harness.memory.repos import OpponentsRepo

__all__ = ["CalcRun", "QuestionLLM", "QuestionOutcome", "answer_question", "tools_for"]

_T = TypeVar("_T", bound=BaseModel)

# Окно, которое модель вправе назвать. Идентификатор сессии в параметрах
# инструмента не появляется: его подставляет код по вопросу «этот вечер или
# вся история», а выдуманный моделью номер сессии молча сузил бы выборку.
WindowArg = Literal["all", "evening"]

# Сколько раз модели дают исправить СВОЙ аргумент, не уронив задачу. Ноль
# (умолчание `Agent(retries=0)` в фасаде) означал бы, что опечатка в значении
# перечисления стоит игроку всего ответа: расчёта при неудачной проверке
# аргумента не случилось, поэтому повтор здесь ничего не пересчитывает
# (`test_a_wrong_argument_value_is_handed_back_to_the_model`).
_TOOL_RETRIES = 1

_ONE_CALC_ONLY = (
    "Расчёт по этому вопросу уже выполнен. Один вопрос — один расчёт: "
    "ответьте по тому, что уже посчитано."
)


class QuestionLLM(Protocol):
    """Та часть фасада `platform.llm.LLM`, которой пользуется маршрутизация.

    Протокол, а не импорт класса: тестам нужен двойник без сети и без
    Postgres, а настоящий `LLM` подходит под него структурно — у него шире
    `purpose` и конкретнее метаданные.
    """

    async def __call__(
        self,
        purpose: Literal["question_answer"],
        schema: type[_T],
        *,
        prompt: str,
        tools: Sequence[Tool[None]] = (),
        trace_id: int,
    ) -> tuple[_T, Any]: ...


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """Чем кончился вопрос: расчёт (если он нашёлся) и слова к нему (если прошли).

    Две независимые величины, и оба `None` — законные состояния:

    * `result is None` — подходящего расчёта в словаре нет, игроку идёт отказ
      со списком того, что посчитать можно;
    * `result` есть, `prose is None` — числа посчитаны, а слова к ним не
      прошли проверку; игрок получает числа без прозы.
    """

    result: CalcResult | None
    prose: str | None


class CalcRun:
    """Расчёты одного хода и их выжимки — то, что копит прогон.

    Не возвращаемое значение, а объект: инструменты замыкаются на нём, и после
    вызова модели вызывающий читает отсюда то, что действительно посчиталось.
    `max_calcs=1` — `/ask`: второй успешный вызов получает отказ
    (`test_the_second_calculation_is_refused`). Агент передаёт 8 (спека агента
    №1, §5): девятый получает отказ (`test_the_ninth_calculation_is_refused`).
    `calls` — журнал каждого вызова, удачного или нет: имя, параметры, выход,
    зона, время (спека агента №1, §8, «Журнал»;
    `test_the_journal_names_calls_checks_and_the_rewrite`).

    `sizes_from` — откуда вправе прийти банк и ставка для `defense_frequency` и
    порога `bet_size`. `None` — `/ask`: размеры из вопроса игрока, не
    проверяются (`test_ask_does_not_check_the_sizes`). Агент передаёт числа
    сводки разбора, реплик игрока и выходов расчётов вечера; к ним добавляются
    выходы расчётов этого хода
    (`test_a_size_from_an_earlier_calculation_of_the_same_turn_is_computed`).
    Размер не оттуда — инструмент не считает и отвечает модели отказом
    (`test_an_invented_pot_is_refused_by_the_tool_and_allows_nothing`,
    `test_a_pot_from_the_hand_summary_is_computed`).
    """

    def __init__(self, max_calcs: int = 1, sizes_from: frozenset[float] | None = None) -> None:
        self.max_calcs = max_calcs
        self.sizes_from = sizes_from
        self.results: list[CalcResult] = []
        self.digests: list[Digest] = []
        self.calls: list[dict[str, Any]] = []

    def unsourced_sizes(self, calc: CalcName, pot_before: int, bet: int) -> str | None:
        """Отказ модели, если банк или ставка не из разрешённого источника; `None` —
        размеры годятся. Отказ пишется в журнал ошибкой вызова."""
        if self.sizes_from is None:
            return None
        known = self.sizes_from.union(*(digest.allowed for digest in self.digests))
        missing = [n for n in (pot_before, bet) if float(n) not in known]
        if not missing:
            return None
        refusal = _UNSOURCED_SIZES.format(numbers=", ".join(str(n) for n in missing))
        self.calls.append({
            "calc": calc.value,
            "params": {"pot_before": pot_before, "bet": bet},
            "error": refusal,
        })
        return refusal

    @property
    def result(self) -> CalcResult | None:
        """Первый расчёт хода; `None` — ни одного удачного вызова инструмента."""
        return self.results[0] if self.results else None

    @property
    def digest(self) -> Digest | None:
        return self.digests[0] if self.digests else None


_UNSOURCED_SIZES = (
    "Расчёт не выполнен: {numbers} нет ни в сводке раздачи, ни в словах игрока, ни в "
    "выходе расчётов. Банк и ставку берите оттуда как есть, не выводите сами; если "
    "их там нет — спросите игрока."
)

_CALC_CAP = "Потолок расчётов на один ответ — {cap}: ответьте по тому, что уже посчитано."

# Зона доверия по инструменту (CLAUDE.md): все шесть расчётов — измерения и
# арифметика, ни один не опирается на угаданный диапазон.
_TOOL_ZONE: dict[CalcName, Zone] = {name: Zone.STRICT for name in CalcName}


async def _perform(
    db: AsyncSession, player_id: int, run_state: CalcRun, params: CalcParams
) -> str | ToolReturn:
    """Выполнить расчёт, если потолок не взят, и вернуть модели выжимку его выхода.

    Ошибку параметра (неизвестная позиция) модель получает текстом и вправе
    позвать инструмент заново: неудавшийся вызов расчётом не считается и
    попытку из потолка не тратит
    (`test_a_bad_parameter_does_not_spend_the_single_calculation`).

    Удачный расчёт возвращается `ToolReturn`: модель видит тот же текст
    выжимки, а метаданные (имя расчёта, разрешённые числа, частота) остаются в
    истории для проверок агента (`test_history_survives_serialization`); что
    модели они не уходят — контракт pydantic-ai (`ToolReturn.metadata`), тестом
    здесь не закреплено.
    """
    params_json = params.model_dump(mode="json")
    if len(run_state.results) >= run_state.max_calcs:
        run_state.calls.append({"calc": params.calc.value, "params": params_json, "refused": True})
        if run_state.max_calcs == 1:
            return _ONE_CALC_ONLY
        return _CALC_CAP.format(cap=run_state.max_calcs)
    started = time.monotonic()
    try:
        result = await run(db, player_id, params)
    except ValueError as exc:
        run_state.calls.append({
            "calc": params.calc.value,
            "params": params_json,
            "error": str(exc),
            "duration_s": round(time.monotonic() - started, 3),
        })
        return f"Расчёт не выполнен: {exc}"
    digest = question_digest(result)
    run_state.results.append(result)
    run_state.digests.append(digest)
    run_state.calls.append({
        "calc": result.calc.value,
        "params": params_json,
        "output": digest.text,
        "zone": _TOOL_ZONE[result.calc].value,
        "duration_s": round(time.monotonic() - started, 3),
    })
    stat = getattr(result, "stat", None)
    return ToolReturn(
        return_value=digest.text,
        metadata={
            "calc": result.calc.value,
            "allowed": sorted(digest.allowed),
            "stat": stat.value if stat is not None else None,
        },
    )


def _window(window: WindowArg, session_id: int) -> Window:
    return Window(session_id=session_id) if window == "evening" else Window()


async def _opponent_id(db: AsyncSession, player_id: int, nick: str) -> int | None:
    """Номер оппонента по нику в руме; `None`, если такого ника игрок не называл.

    Ник, а не номер, потому что номер модели неоткуда взять правильным, а
    выдуманный номер дал бы выборку чужого оппонента или пустую.
    """
    for record in await OpponentsRepo(db).list_for_player(player_id):
        if record.nick.casefold() == nick.casefold():
            return record.opponent_id
    return None


def _unknown_opponent(nick: str) -> str:
    return (
        f"Оппонента с ником {nick!r} игрок не называл. Привязка оппонента к его "
        f"местам делается командой /alias, и без неё считается поле."
    )


def tools_for(
    db: AsyncSession, player_id: int, session_id: int, run_state: CalcRun
) -> list[Tool[None]]:
    """Шесть инструментов — по одному на имя из `CalcName`, и ни одного сверх.

    Имена инструментов совпадают с именами словаря
    (`test_the_tool_set_covers_every_calculation_name_exactly_once`): набор,
    из которого выбирает модель, и набор, который умеет считать код, — один и
    тот же список, а не две копии, способные разойтись.
    """

    async def hero_frequency(
        stat: FrequencyStat, position: str | None = None, window: WindowArg = "all"
    ) -> str | ToolReturn:
        """Частота самого игрока по его разобранным раздачам.

        Args:
            stat: какая частота измеряется.
            position: позиция игрока за столом, например BTN; без неё — все позиции.
            window: `evening` — текущий вечер, `all` — вся история разборов.
        """
        return await _perform(
            db,
            player_id,
            run_state,
            HeroFrequencyParams(stat=stat, position=position, window=_window(window, session_id)),
        )

    async def opponent_frequency(
        stat: FrequencyStat,
        opponent: str | None = None,
        position: str | None = None,
        window: WindowArg = "all",
    ) -> str | ToolReturn:
        """Частота оппонента: названного ником либо поля целиком.

        Args:
            stat: какая частота измеряется.
            opponent: ник в руме, если спрашивают про конкретного; без него — поле.
            position: позиция оппонента за столом; без неё — все позиции.
            window: `evening` — текущий вечер, `all` — вся история разборов.
        """
        opponent_id: int | None = None
        if opponent is not None:
            opponent_id = await _opponent_id(db, player_id, opponent)
            if opponent_id is None:
                return _unknown_opponent(opponent)
        return await _perform(
            db,
            player_id,
            run_state,
            OpponentFrequencyParams(
                stat=stat,
                opponent_id=opponent_id,
                position=position,
                window=_window(window, session_id),
            ),
        )

    async def coverage(
        street: Street | None = None,
        spot: SpotKind | None = None,
        position: str | None = None,
        window: WindowArg = "all",
    ) -> str | ToolReturn:
        """Сколько точек решения разобрано, у скольких посчитана цена и какова сумма.

        Args:
            street: улица точки решения; без неё — все улицы.
            spot: тип спота; без него — все споты.
            position: позиция игрока; без неё — все позиции.
            window: `evening` — текущий вечер, `all` — вся история разборов.
        """
        return await _perform(
            db,
            player_id,
            run_state,
            CoverageParams(
                filter=PointFilter(
                    street=street,
                    spot=spot,
                    position=position,
                    window=_window(window, session_id),
                )
            ),
        )

    async def leaks(window: WindowArg = "all") -> str | ToolReturn:
        """Типы повторяющихся расхождений с ценой каждого.

        Args:
            window: `evening` — текущий вечер, `all` — вся история разборов.
        """
        return await _perform(
            db, player_id, run_state, LeaksParams(window=_window(window, session_id))
        )

    async def defense_frequency(pot_before: int, bet: int) -> str | ToolReturn:
        """Требуемая частота защиты против ставки — арифметика, данных не требует.

        Args:
            pot_before: банк в фишках ДО ставки соперника.
            bet: размер ставки соперника в фишках.
        """
        refusal = run_state.unsourced_sizes(CalcName.DEFENSE_FREQUENCY, pot_before, bet)
        if refusal is not None:
            return refusal
        return await _perform(
            db, player_id, run_state, DefenseParams(pot_before=pot_before, bet=bet)
        )

    async def frequency_vs_threshold(
        stat: FrequencyStat,
        subject: Subject = Subject.HERO,
        threshold_source: Literal["bet_size", "field"] = "field",
        pot_before: int | None = None,
        bet: int | None = None,
        side: ThresholdSide = ThresholdSide.FOLD,
        opponent: str | None = None,
        position: str | None = None,
        window: WindowArg = "all",
    ) -> str | ToolReturn:
        """Частота против порога: утверждение либо число недостающих наблюдений.

        Args:
            stat: какая частота измеряется.
            subject: чья частота — `hero`, `opponent` или `field`.
            threshold_source: `bet_size` — арифметика от размера ставки,
                `field` — измеренная частота поля.
            pot_before: банк в фишках до ставки; нужен при `bet_size`.
            bet: размер ставки в фишках; нужен при `bet_size`.
            side: какая из двух арифметических частот берётся порогом.
            opponent: ник в руме; нужен при `subject=opponent`.
            position: позиция за столом; без неё — все позиции.
            window: `evening` — текущий вечер, `all` — вся история разборов.
        """
        opponent_id: int | None = None
        if subject is Subject.OPPONENT:
            if opponent is None:
                return (
                    "Для частоты конкретного оппонента нужен его ник в руме; "
                    "без ника считается поле (`subject=field`)."
                )
            opponent_id = await _opponent_id(db, player_id, opponent)
            if opponent_id is None:
                return _unknown_opponent(opponent)
        if threshold_source == "bet_size":
            if pot_before is None or bet is None:
                return (
                    "Порог от размера ставки требует банка до ставки и самой ставки "
                    "в фишках."
                )
            refusal = run_state.unsourced_sizes(
                CalcName.FREQUENCY_VS_THRESHOLD, pot_before, bet
            )
            if refusal is not None:
                return refusal
            threshold: BetSizeThreshold | FieldThreshold = BetSizeThreshold(
                pot_before=pot_before, bet=bet, side=side
            )
        else:
            threshold = FieldThreshold()
        return await _perform(
            db,
            player_id,
            run_state,
            ThresholdParams(
                subject=subject,
                stat=stat,
                opponent_id=opponent_id,
                position=position,
                window=_window(window, session_id),
                threshold=threshold,
            ),
        )

    by_name: dict[CalcName, Any] = {
        CalcName.HERO_FREQUENCY: hero_frequency,
        CalcName.OPPONENT_FREQUENCY: opponent_frequency,
        CalcName.COVERAGE: coverage,
        CalcName.LEAKS: leaks,
        CalcName.DEFENSE_FREQUENCY: defense_frequency,
        CalcName.FREQUENCY_VS_THRESHOLD: frequency_vs_threshold,
    }
    return [
        Tool(function, name=name.value, max_retries=_TOOL_RETRIES)
        for name, function in by_name.items()
    ]


async def answer_question(
    db: AsyncSession,
    llm: QuestionLLM,
    *,
    player_id: int,
    session_id: int,
    question: str,
    trace_id: int,
) -> QuestionOutcome:
    """Один вызов модели с инструментами: что посчитано и что об этом сказано.

    Модель зовётся один раз; повторов при непрошедшем тексте нет — числа уже
    посчитаны, и платить второй раз за слова к ним незачем (то же решение, что
    у `verdict_text`).
    """
    run_state = CalcRun()
    tools = tools_for(db, player_id, session_id, run_state)
    draft, _meta = await llm(
        "question_answer",
        QuestionDraft,
        prompt=question_prompt(question),
        tools=tools,
        trace_id=trace_id,
    )
    if run_state.result is None or run_state.digest is None:
        return QuestionOutcome(result=None, prose=None)
    try:
        check_answer(draft.answer, run_state.digest.allowed)
    except UnfaithfulText:
        return QuestionOutcome(result=run_state.result, prose=None)
    return QuestionOutcome(result=run_state.result, prose=draft.answer.strip())
