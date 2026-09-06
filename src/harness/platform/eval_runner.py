"""Прогон evals по настоящей модели — руками, не в наборе тестов.

Три подкоманды, по одной на каждый вход модели в системе плюс зрение:
`verdict` — текст разбора одной раздачи, `tournament --hh FILE` — рассказ по
турниру, `vision --cases DIR` — чтение скриншотов (этаж 2 EVALS.md, самый
важный и дорогой). Вторая появилась по ревью: у второго входа модели eval не
было вовсе, а именно там модель написала игроку про «раздачи без явных ошибок»
— утверждение о том, чего расчёт не судил.

**Зрение сравнивается с текстом рума, а не с разметкой руками.** Если рука со
скрина есть в hand history, eval-кейс собирается кодом: экран — вход, текст
рума — ожидаемый выход (реестр, «G2 ослабляется»). Номер раздачи, прочитанный
неверно, просто не найдётся в файле — и это первая проверка, бесплатная.

`uv run python -m harness.platform.eval_runner verdict` — этаж 3 EVALS.md:
берёт кейсы (готовые `AnalysisResult`), зовёт `LLM_VERDICT_MODEL` через тот же
фасад, что и прод (`platform/llm.py`: лимитер, строки `llm_calls`, ретраи), и
прогоняет по ответу три проверки из `evals/verdict/checks.py`. Печатает текст
модели целиком и итог по каждой проверке; ненулевой код возврата — хотя бы один
кейс не прошёл.

**Обязателен при смене `LLM_VERDICT_MODEL` или промпта вердикта.** Тесты этого
не заметят: они ходят к двойнику модели, а не к модели.

**Почему нужен Postgres.** `llm_calls.trace_id` — FK NOT NULL, и фасад пишет
строку до каждого обращения к провайдеру; лимитер тоже считает по этой таблице.
Прогон поэтому заводит служебную цепочку `player → session → job → trace` в базе
из `DATABASE_URL` и работает как обычная задача воркера. Служебный игрок узнаётся
по `tg_user_id = 0` (настоящих Телеграм-аккаунтов с таким id не бывает).

**Два источника кейсов.**

* `--cases DIR` (умолчание — `evals/verdict/cases`): синтетические разборы в
  JSON. Они и лежат в репозитории: настоящих рук игрока в публичной истории быть
  не должно (политика публикации), а синтетика проверяет ровно то же самое —
  умеет ли модель излагать чужие числа, не выдумывая своих.
* `--hh FILE --hands N`: N раздач с вердиктом из настоящего файла hand history.
  Файл берётся с диска и в репозиторий не попадает; это способ прогнать eval на
  живых данных перед сменой модели.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

from sqlalchemy import select

from harness.analysis import analyze_hand
from harness.analysis.scan import scan_tournament
from harness.analysis.tournament import tournament_report
from harness.contracts import (
    AnalysisResult,
    CanonicalHand,
    EnrichedHand,
    PlayerState,
    TournamentReport,
    ValidationStatus,
    VisionReading,
)
from harness.engine import enrich
from harness.explanation.tournament_text import tournament_draft
from harness.explanation.verdict_text import verdict_draft
from harness.memory.models import Job, LlmCall, Trace, async_session_factory
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file
from harness.parsers.vision_adapter import vision_extract
from harness.platform.config import Config
from harness.platform.llm import LLM

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CASES = _REPO_ROOT / "evals" / "verdict" / "cases"
_CHECKS_PATH = _REPO_ROOT / "evals" / "verdict" / "checks.py"
_DEFAULT_VISION_CASES = _REPO_ROOT / "evals" / "vision" / "cases"

# Допуск сверки сумм в больших блайндах: экран печатает два знака и ОБРЕЗАЕТ,
# и на восьмерых накопленная обрезка не выходит за сотые доли ББ.
_AMOUNT_TOLERANCE_BB = 0.02
_POT_TOLERANCE_BB = 0.1
# Анте — доли ББ, и допуск стеков рядом с ним слишком груб: пул, поделённый на
# ДЕВЯТЬ игроков вместо восьми, отличается от верного на 0.013 ББ и прошёл бы.
# Само анте код не читает, а считает, поэтому обрезки экрана в нём нет вовсе.
_ANTE_TOLERANCE_BB = 0.005

# Поля, цена ошибки в которых выше остальных (EVALS.md, этаж 2): по ним метрика
# считается отдельно, потому что средняя точность по всем полям их растворяет.
_CRITICAL_FIELDS = frozenset({"hand_no", "hero_cards", "stacks", "actions"})

# Служебный игрок прогона: `players.tg_user_id` уникален, и 0 не может
# принадлежать живому аккаунту Телеграма.
_EVAL_TG_USER_ID = 0


def _load_checks() -> ModuleType:
    """Импорт `evals/verdict/checks.py` по пути: `evals/` — не пакет и им не станет.

    Каталог evals лежит рядом с кодом, а не внутри него, потому что в образ он не
    едет (там нечего проверять) и в `src` его импортировать неоткуда. Отсюда
    загрузка по файлу — с внятным отказом, если файла нет.
    """
    if not _CHECKS_PATH.exists():
        raise SystemExit(f"нет файла проверок {_CHECKS_PATH} — прогон evals невозможен")
    spec = importlib.util.spec_from_file_location("evals.verdict.checks", _CHECKS_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover — путь существует
        raise SystemExit(f"не удалось загрузить {_CHECKS_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Модуль обязан лежать в `sys.modules` ДО исполнения: `@dataclass` с
    # отложенными аннотациями (`from __future__ import annotations`) разрешает их
    # через `sys.modules[cls.__module__]`, и без регистрации падает на первом же
    # классе.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cases_from_dir(directory: Path) -> list[tuple[str, AnalysisResult]]:
    """Кейсы из JSON-файлов каталога, по имени файла."""
    files = sorted(directory.glob("*.json"))
    if not files:
        raise SystemExit(f"в {directory} нет ни одного кейса (*.json)")
    return [
        (path.stem, AnalysisResult.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        for path in files
    ]


def _enriched_from_hh(source: Path) -> list[EnrichedHand]:
    """Раздачи файла, прошедшие движок и валидатор, — общий вход обоих прогонов."""
    return [
        enriched
        for enriched in (
            enrich(normalize(raw))
            for raw in parse_file(source.read_text(encoding="utf-8"), str(source))
        )
        if enriched.verdict.status is not ValidationStatus.REJECT
    ]


def _report_from_hh(source: Path) -> TournamentReport:
    """Отчёт по турниру из файла — тот же путь, что у воркера: скан, затем отчёт."""
    enriched = _enriched_from_hh(source)
    if not enriched:
        raise SystemExit(f"в {source} не нашлось ни одной пригодной раздачи")
    summary = scan_tournament(enriched)
    return tournament_report(
        enriched, summary, player_tournaments=[[en.hand for en in enriched]]
    )


def _cases_from_hh(source: Path, limit: int) -> list[tuple[str, AnalysisResult]]:
    """Первые `limit` раздач файла, у которых есть хоть одна судимая точка.

    Раздача без вердикта кейсом быть не может: излагать нечего, и модель на ней
    не вызывается вовсе (`verdict_text`).
    """
    cases: list[tuple[str, AnalysisResult]] = []
    for enriched in _enriched_from_hh(source):
        result = analyze_hand(enriched)
        if not result.ranked:
            continue
        cases.append((f"hh:{result.hand_no}", result))
        if len(cases) == limit:
            break
    if not cases:
        raise SystemExit(f"в {source} не нашлось раздач с вердиктом")
    return cases


# --- vision (этаж 2 EVALS.md) -------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldResult:
    """Одно сравненное поле: что прочитано с экрана и что пишет рум."""

    field: str
    matched: bool
    read: str = ""
    expected: str = ""

    @property
    def critical(self) -> bool:
        return self.field in _CRITICAL_FIELDS


def _bb(amount: int, bb: int) -> float:
    return amount / bb if bb else 0.0


def _by_position(hand: CanonicalHand) -> dict[str, PlayerState]:
    return {player.position: player for player in hand.players}


def _actions_by_position(hand: CanonicalHand) -> list[tuple[str, str, str, float]]:
    """Действия как «улица, позиция, вид, сумма в ББ» — вид сравнения, общий для
    экрана и текста рума: меток игроков у них разные, а позиции одни и те же.
    """
    positions = {player.label: player.position for player in hand.players}
    return [
        (
            action.street.value,
            positions.get(action.label, "?"),
            action.kind.value,
            _bb(action.committed_after, hand.bb),
        )
        for action in hand.actions
    ]


def _actions_match(read: CanonicalHand, truth: CanonicalHand) -> bool:
    """Совпали ли действия — с тем же допуском на обрезку, что и суммы.

    Точное сравнение сумм здесь измеряло бы не чтение, а округление экрана:
    вложенные 14.6268 ББ он печатает как «14.62», и посчитанное по тексту рума
    14.63 расходится с прочитанным на сотую — на каждой второй руке. Первый
    прогон датасета так и вышел: шесть «расхождений» из девятнадцати оказались
    этой сотой, а не ошибкой модели.
    """
    left, right = _actions_by_position(read), _actions_by_position(truth)
    if len(left) != len(right):
        return False
    return all(
        a[:3] == b[:3] and abs(a[3] - b[3]) <= _AMOUNT_TOLERANCE_BB
        for a, b in zip(left, right, strict=True)
    )


def _cards_of(hand: CanonicalHand, label: str) -> list[str]:
    seen = hand.dealt.get(label, [])
    if seen:
        return sorted(seen)
    for entry in hand.showdowns:
        if entry.label == label:
            return sorted(entry.cards)
    return []


def compare_reading(read: CanonicalHand, truth: CanonicalHand) -> list[FieldResult]:
    """Сравнить прочитанное с экрана с тем, что записал рум, — по типам полей.

    Сравнение идёт ПО ПОЗИЦИЯМ, а не по меткам игроков: на экране ники, в тексте
    рума анонимные хеши, и единственное, что у них общее, — раскладка от кнопки.
    Отсюда же следует, что сместившаяся рассадка проявится как расхождение
    стеков: тот же приём, что и у контрольных сумм адаптера, — сверка двух
    независимых прочтений одного факта.

    Суммы сравниваются в больших блайндах: экран печатает только их, фишек на
    нём нет вовсе.
    """
    results = [
        FieldResult(
            field="hand_no",
            matched=read.hand_no == truth.hand_no,
            read=read.hand_no,
            expected=truth.hand_no,
        ),
        FieldResult(
            field="players",
            matched=len(read.players) == len(truth.players),
            read=str(len(read.players)),
            expected=str(len(truth.players)),
        ),
        FieldResult(
            field="blinds",
            matched=abs(_bb(read.sb, read.bb) - _bb(truth.sb, truth.bb)) <= _AMOUNT_TOLERANCE_BB,
            read=f"{_bb(read.sb, read.bb):.2f}",
            expected=f"{_bb(truth.sb, truth.bb):.2f}",
        ),
        FieldResult(
            field="ante",
            matched=(
                abs(_bb(read.ante, read.bb) - _bb(truth.ante, truth.bb)) <= _ANTE_TOLERANCE_BB
            ),
            read=f"{_bb(read.ante, read.bb):.3f}",
            expected=f"{_bb(truth.ante, truth.bb):.3f}",
        ),
    ]

    read_board = [card for cards in read.boards.values() for card in cards]
    truth_board = [card for cards in truth.boards.values() for card in cards]
    results.append(
        FieldResult(
            field="board",
            matched=read_board == truth_board,
            read=" ".join(read_board),
            expected=" ".join(truth_board),
        )
    )

    hero_read = _cards_of(read, read.hero_label)
    hero_truth = _cards_of(truth, truth.hero_label)
    results.append(
        FieldResult(
            field="hero_cards",
            matched=hero_read == hero_truth,
            read=" ".join(hero_read),
            expected=" ".join(hero_truth),
        )
    )

    read_seats, truth_seats = _by_position(read), _by_position(truth)
    mismatched_stacks: list[str] = []
    for position, player in truth_seats.items():
        expected = _bb(player.stack, truth.bb)
        seat = read_seats.get(position)
        if seat is None:
            mismatched_stacks.append(f"{position}: места нет, рум {expected:.2f}")
            continue
        got = _bb(seat.stack, read.bb)
        if abs(got - expected) > _AMOUNT_TOLERANCE_BB:
            mismatched_stacks.append(f"{position}: {got:.2f} против {expected:.2f}")
    results.append(
        FieldResult(
            field="stacks",
            matched=not mismatched_stacks,
            read="; ".join(mismatched_stacks),
            expected=f"{len(truth_seats)} мест",
        )
    )

    results.append(
        FieldResult(
            field="actions",
            matched=_actions_match(read, truth),
            read=str([(a[0], a[1], a[2], round(a[3], 2)) for a in _actions_by_position(read)]),
            expected=str([(a[0], a[1], a[2], round(a[3], 2)) for a in _actions_by_position(truth)]),
        )
    )

    read_shown = {
        entry.label: sorted(entry.cards) for entry in read.showdowns if len(entry.cards) == 2
    }
    truth_positions = {player.label: player.position for player in truth.players}
    read_positions = {player.label: player.position for player in read.players}
    shown_by_position = {read_positions.get(label, "?"): cards for label, cards in read_shown.items()}
    truth_by_position = {
        truth_positions.get(entry.label, "?"): sorted(entry.cards)
        for entry in truth.showdowns
        if len(entry.cards) == 2
    }
    common = set(shown_by_position) & set(truth_by_position)
    results.append(
        FieldResult(
            field="showdown",
            matched=all(shown_by_position[p] == truth_by_position[p] for p in common),
            read=str(sorted(shown_by_position.items())),
            expected=str(sorted(truth_by_position.items())),
        )
    )
    return results


def _pot_result(read: EnrichedHand, truth: CanonicalHand) -> FieldResult:
    """Банк, ПОСЧИТАННЫЙ движком по прочитанному, против итога рума.

    Считает движок, а не сложение с экрана: сойтись обязан именно тот банк, за
    который дальше судится решение, а не число, напечатанное рядом с фишками.
    """
    engine_bb = _bb(read.report.final_pot, read.hand.bb)
    truth_bb = _bb(truth.summary.total_pot if truth.summary else 0, truth.bb)
    return FieldResult(
        field="pot",
        matched=abs(engine_bb - truth_bb) <= _POT_TOLERANCE_BB,
        read=f"{engine_bb:.2f}",
        expected=f"{truth_bb:.2f}",
    )


def _truth_by_hand_no(source: Path, hand_no: str) -> CanonicalHand | None:
    """Эталонная рука из текста рума по номеру. `None` — номер прочитан неверно.

    Это первая и самая дешёвая проверка чтения: выдуманного номера в файле не
    существует, и дальше сравнивать уже нечего.
    """
    for raw in parse_file(source.read_text(encoding="utf-8"), str(source)):
        if raw.hand_no == hand_no:
            return normalize(raw)
    return None


async def _eval_trace(session_factory) -> int:
    """Служебная цепочка FK для `llm_calls.trace_id` — см. модульный докстринг."""
    async with session_factory() as session:
        player = await PlayersRepo(session).get_or_create(tg_user_id=_EVAL_TG_USER_ID)
        player_session = await SessionsRepo(session).active_or_create(player.id)
        job = Job(
            type="eval_run",
            player_id=player.id,
            session_id=player_session.id,
            payload={"kind": "verdict"},
        )
        session.add(job)
        await session.flush()
        trace = Trace(job_id=job.id)
        session.add(trace)
        await session.commit()
        return trace.id


async def _call_costs(session_factory, trace_id: int) -> list[tuple[str, int, int, float | None, int]]:
    """Что стоил прогон: строки `llm_calls` этого трейса — модель, токены, деньги."""
    async with session_factory() as session:
        rows = await session.execute(
            select(
                LlmCall.model,
                LlmCall.tokens_in,
                LlmCall.tokens_out,
                LlmCall.cost,
                LlmCall.latency_ms,
            ).where(LlmCall.trace_id == trace_id)
        )
        return [tuple(row) for row in rows.all()]


async def _open_llm(cfg: Config) -> tuple[LLM, int, Any]:
    """Фасад модели, служебный трейс и фабрика сессий — общее начало обоих прогонов."""
    session_factory = async_session_factory(cfg.database_url)
    llm = LLM(cfg, session_factory)
    trace_id = await _eval_trace(session_factory)
    return llm, trace_id, session_factory


async def _print_costs(session_factory: Any, trace_id: int) -> None:
    """Что стоил прогон — одной строкой, из `llm_calls`.

    `tokens_in`/`tokens_out`/`cost` пусты у строк, вставленных ПЕРЕД вызовом и
    не дошедших до ответа (`llm_calls` пишется до обращения к провайдеру —
    `platform/llm.py`, пункт 2). Считать их нулями верно: неотвеченная попытка
    токенов не потратила; скрыть саму строку было бы неправдой о нагрузке.
    """
    costs = await _call_costs(session_factory, trace_id)
    tokens_in = sum(row[1] or 0 for row in costs)
    tokens_out = sum(row[2] or 0 for row in costs)
    # `Decimal`, а не float: `llm_calls.cost` — NUMERIC, и сложение с нулём-float
    # роняет прогон ПОСЛЕ того, как за него уже заплачено (найдено на живом
    # прогоне vision-датасета — до него колонка всегда была пустой).
    money = float(sum((row[3] or Decimal(0)) for row in costs))
    per_call = len(costs) or 1
    print(
        f"Вызовов модели: {len(costs)}; токенов на вход {tokens_in}, на выход "
        f"{tokens_out}; оценка стоимости ${money:.6f} за прогон, "
        f"${money / per_call:.6f} за вызов (оценка провайдер-слоя, не счёт)."
    )


async def _run_tournament(args: argparse.Namespace) -> int:
    """Прогон второго входа модели — рассказа по турниру (ревью, раздел B).

    Кейс здесь всегда один и всегда из файла: синтетического отчёта по турниру,
    осмысленного для проверки, не существует — его пришлось бы выдумывать
    целиком, включая траекторию стека и разбиение фишек.
    """
    checks = _load_checks()
    report = _report_from_hh(Path(args.hh))
    cfg = Config.from_env()
    llm, trace_id, session_factory = await _open_llm(cfg)

    print(
        f"Модель вердикта: {cfg.llm_verdict_model}; турнир из {args.hh} "
        f"({report.hands_total} раздач); трейс {trace_id}\n"
    )
    story, digest = await tournament_draft(llm, report, trace_id=trace_id)
    for index, paragraph in enumerate(story.paragraphs, start=1):
        print(f"  [{index}] {paragraph}")
    print()
    results = checks.run_story_checks(story, digest.allowed)
    for check in results:
        mark = "OK  " if check.passed else "ПРОВАЛ"
        print(f"  {mark} {check.name}{f': {check.detail}' if check.detail else ''}")
    print()
    await _print_costs(session_factory, trace_id)
    print(f"Провалено проверок: {sum(not c.passed for c in results)} из {len(results)}.")
    return 1 if any(not check.passed for check in results) else 0


async def _run_verdict(args: argparse.Namespace) -> int:
    """Прогон текста разбора: кейсы из каталога либо из настоящего файла раздач."""
    checks = _load_checks()
    cases = (
        _cases_from_hh(Path(args.hh), args.hands)
        if args.hh
        else _cases_from_dir(Path(args.cases))
    )

    cfg = Config.from_env()
    llm, trace_id, session_factory = await _open_llm(cfg)

    print(f"Модель вердикта: {cfg.llm_verdict_model}; кейсов: {len(cases)}; трейс {trace_id}\n")
    failed = 0
    for name, res in cases:
        draft, digest = await verdict_draft(llm, res, trace_id=trace_id)
        results = checks.run_checks(draft, res, digest.allowed)
        print(f"=== {name} (точек: {len(res.ranked)})")
        for point in draft.points:
            print(f"  [{point.dp_index}] {point.text}")
        print(f"  вывод: {draft.summary}")
        for check in results:
            mark = "OK  " if check.passed else "ПРОВАЛ"
            print(f"  {mark} {check.name}{f': {check.detail}' if check.detail else ''}")
        failed += any(not check.passed for check in results)
        print()

    await _print_costs(session_factory, trace_id)
    print(f"Кейсов провалено: {failed} из {len(cases)}.")
    return 1 if failed else 0


def _vision_cases(directory: Path) -> list[dict[str, Any]]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise SystemExit(f"в {directory} нет ни одного кейса (*.json)")
    return [
        {"name": path.stem, **json.loads(path.read_text(encoding="utf-8"))} for path in files
    ]


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else _REPO_ROOT / path


class _CannedLLM:
    """Двойник фасада для синтетических кейсов: отдаёт готовое чтение, не платя.

    Синтетика держит МЕХАНИКУ раннера — разбор кейса, сравнение по полям, счёт
    метрик — и держит её бесплатно. Качество чтения она не меряет и не может:
    для этого нужен настоящий экран и настоящая модель.
    """

    def __init__(self, reading: VisionReading) -> None:
        self.reading = reading

    async def __call__(self, purpose, schema, *, prompt, images=(), trace_id):
        return self.reading, type("Meta", (), {"model": "canned"})()


async def _read_case(
    case: dict[str, Any], llm: LLM, *, nickname: str, trace_id: int, fallback: bool
):
    """Прочитать экран кейса — настоящей моделью либо заготовленным чтением."""
    image_path = _resolve(case["image"]) if case.get("image") else None
    image = image_path.read_bytes() if image_path is not None else b""
    reader: Any = llm
    if case.get("reading") is not None:
        reader = _CannedLLM(VisionReading.model_validate(case["reading"]))
    return await vision_extract(
        reader,
        image,
        gg_nickname=case.get("gg_nickname") or nickname,
        trace_id=trace_id,
        source_ref=case["name"],
        fallback_available=fallback and case.get("reading") is None,
    )


def _score_case(case: dict[str, Any], outcome: Any) -> dict[str, Any]:
    """Итог одного кейса: отказ, ненайденный номер руки или сравнение по полям."""
    row: dict[str, Any] = {
        "name": case["name"],
        "kind": case.get("kind", "hand"),
        "hops": [hop.role for hop in outcome.hops],
        "failed_checks": [check.name for check in outcome.failed],
        "escalated": outcome.escalate,
        "fields": [],
    }
    if case.get("kind") == "refusal":
        row["refused"] = outcome.raw is None
        row["refusal_reason"] = outcome.refusal or ""
        return row
    if outcome.raw is None:
        row["error"] = f"модель отказалась читать раздачу: {outcome.refusal}"
        return row

    read = normalize(outcome.raw)
    # Прочитанная рука кладётся в отчёт целиком: пересчитать метрики по ней
    # можно бесплатно, а повторить прогон — только за деньги.
    row["raw"] = outcome.raw.model_dump(mode="json")
    row["hand_no"] = read.hand_no
    truth = _truth_by_hand_no(_resolve(case["hh"]), read.hand_no)
    if truth is None:
        row["fields"] = [
            {"field": "hand_no", "matched": False, "read": read.hand_no, "expected": "нет в файле"}
        ]
        row["error"] = "номер раздачи не найден в тексте рума"
        return row

    enriched = enrich(read)
    row["validator"] = enriched.verdict.status.value
    row["validator_reasons"] = enriched.verdict.reasons
    results = [*compare_reading(read, truth), _pot_result(enriched, truth)]
    row["fields"] = [
        {
            "field": r.field,
            "matched": r.matched,
            "critical": r.critical,
            "read": r.read,
            "expected": r.expected,
        }
        for r in results
    ]
    return row


def _vision_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Метрики этажа 2: по типам полей, отдельно критичные, доля без эскалации."""
    by_field: dict[str, list[bool]] = {}
    for row in rows:
        for field in row["fields"]:
            by_field.setdefault(field["field"], []).append(bool(field["matched"]))

    hands = [row for row in rows if row.get("kind", "hand") == "hand"]
    refusals = [row for row in rows if row.get("kind") == "refusal"]
    critical = [
        matched
        for row in hands
        for field in row["fields"]
        if field.get("critical")
        for matched in [bool(field["matched"])]
    ]
    return {
        "hands": len(hands),
        "refusals": len(refusals),
        "by_field": {
            name: {"matched": sum(values), "total": len(values)}
            for name, values in sorted(by_field.items())
        },
        "critical": {"matched": sum(critical), "total": len(critical)},
        "clean_pass": sum(
            1 for row in hands if row.get("validator") == "pass" and not row["escalated"]
        ),
        "refused": sum(1 for row in refusals if row.get("refused")),
    }


async def _run_vision(args: argparse.Namespace) -> int:
    """Прогон зрения: скрин -> `RawHand` -> сверка с текстом рума по полям.

    Эталон берётся из hand history по номеру раздачи, прочитанному с экрана.
    Разметки руками это не требует — и именно поэтому прогон воспроизводим:
    заново он даст те же ожидания, а не чью-то память о них.
    """
    cases = _vision_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]
    cfg = Config.from_env()
    llm, trace_id, session_factory = await _open_llm(cfg)
    fallback = bool(cfg.llm_vision_fallback_model)

    print(
        f"Модель зрения: {cfg.llm_vision_model}"
        f"{' -> ' + cfg.llm_vision_fallback_model if fallback else ' (каскада нет)'}; "
        f"кейсов: {len(cases)}; трейс {trace_id}\n"
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            outcome = await _read_case(
                case, llm, nickname=args.nickname, trace_id=trace_id, fallback=fallback
            )
            row = _score_case(case, outcome)
        except Exception as exc:  # noqa: BLE001 — один упавший кейс не отменяет прогон
            # Трассировка целиком: кейс падает на ЧУЖИХ данных (настоящий экран),
            # и один `repr` сообщает, что сломалось, но не где — а воспроизвести
            # прогон стоит денег, поэтому второй попытки на диагностику нет.
            row = {"name": case["name"], "kind": case.get("kind", "hand"), "fields": [],
                   "hops": [], "failed_checks": [], "escalated": False, "error": repr(exc),
                   "traceback": traceback.format_exc()}
        rows.append(row)
        mark = "OK  " if not row.get("error") and all(
            f["matched"] for f in row["fields"]
        ) else "РАЗН"
        if row.get("kind") == "refusal":
            mark = "OK  " if row.get("refused") else "ПРОВАЛ"
        print(f"{mark} {row['name']}: ступеней {len(row['hops'])}, "
              f"непройденных сверок {row['failed_checks']}"
              f"{', ' + row['error'] if row.get('error') else ''}")
        for field in row["fields"]:
            if not field["matched"]:
                print(f"       {field['field']}: прочитано {field['read']!r}, "
                      f"рум {field['expected']!r}")

    metrics = _vision_metrics(rows)
    print("\nТочность по типам полей:")
    for name, stat in metrics["by_field"].items():
        print(f"  {name:12} {stat['matched']}/{stat['total']}")
    print(
        f"\nКритичные поля: {metrics['critical']['matched']}/{metrics['critical']['total']}; "
        f"без эскалации прошло {metrics['clean_pass']}/{metrics['hands']}; "
        f"отказов на не-руках {metrics['refused']}/{metrics['refusals']}."
    )
    # Отчёт пишется ДО подсчёта стоимости, а не после: прогон уже оплачен, и
    # любая поломка в печати итогов не имеет права стоить результат (первый
    # полный прогон датасета потерялся именно так).
    if args.out:
        Path(args.out).write_text(
            json.dumps({"metrics": metrics, "cases": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Подробности: {args.out}")
    await _print_costs(session_factory, trace_id)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_runner", description="прогон evals по модели")
    sub = parser.add_subparsers(dest="suite", required=True)
    verdict = sub.add_parser("verdict", help="этаж 3: верность изложения вердикта")
    verdict.add_argument("--cases", default=str(_DEFAULT_CASES), help="каталог с кейсами (*.json)")
    verdict.add_argument("--hh", default=None, help="файл hand history вместо каталога кейсов")
    verdict.add_argument("--hands", type=int, default=3, help="сколько раздач взять из --hh")
    tournament = sub.add_parser("tournament", help="этаж 3: верность рассказа по турниру")
    tournament.add_argument("--hh", required=True, help="файл hand history одного турнира")
    vision = sub.add_parser("vision", help="этаж 2: точность чтения скриншотов")
    vision.add_argument("--cases", default=str(_DEFAULT_VISION_CASES), help="каталог кейсов")
    vision.add_argument("--nickname", default="", help="ник игрока в руме (в промпт НЕ идёт)")
    vision.add_argument("--limit", type=int, default=0, help="взять только первые N кейсов")
    vision.add_argument("--out", default=None, help="куда записать подробный отчёт (JSON)")
    args = parser.parse_args(argv)
    if args.suite == "tournament":
        return asyncio.run(_run_tournament(args))
    if args.suite == "vision":
        return asyncio.run(_run_vision(args))
    return asyncio.run(_run_verdict(args))


if __name__ == "__main__":  # pragma: no cover — точка входа CLI
    sys.exit(main())
