"""Прогон evals по настоящей модели — руками, не в наборе тестов.

Две подкоманды, по одной на каждый вход модели в системе: `verdict` — текст
разбора одной раздачи, `tournament --hh FILE` — рассказ по турниру. Вторая
появилась по ревью: у второго входа модели eval не было вовсе, а именно там
модель написала игроку про «раздачи без явных ошибок» — утверждение о том,
чего расчёт не судил.

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
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from sqlalchemy import select

from harness.analysis import analyze_hand
from harness.analysis.scan import scan_tournament
from harness.analysis.tournament import tournament_report
from harness.contracts import AnalysisResult, EnrichedHand, TournamentReport, ValidationStatus
from harness.engine import enrich
from harness.explanation.tournament_text import tournament_draft
from harness.explanation.verdict_text import verdict_draft
from harness.memory.models import Job, LlmCall, Trace, async_session_factory
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file
from harness.platform.config import Config
from harness.platform.llm import LLM

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CASES = _REPO_ROOT / "evals" / "verdict" / "cases"
_CHECKS_PATH = _REPO_ROOT / "evals" / "verdict" / "checks.py"

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
    money = sum(row[3] or 0.0 for row in costs)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_runner", description="прогон evals по модели")
    sub = parser.add_subparsers(dest="suite", required=True)
    verdict = sub.add_parser("verdict", help="этаж 3: верность изложения вердикта")
    verdict.add_argument("--cases", default=str(_DEFAULT_CASES), help="каталог с кейсами (*.json)")
    verdict.add_argument("--hh", default=None, help="файл hand history вместо каталога кейсов")
    verdict.add_argument("--hands", type=int, default=3, help="сколько раздач взять из --hh")
    tournament = sub.add_parser("tournament", help="этаж 3: верность рассказа по турниру")
    tournament.add_argument("--hh", required=True, help="файл hand history одного турнира")
    args = parser.parse_args(argv)
    if args.suite == "tournament":
        return asyncio.run(_run_tournament(args))
    return asyncio.run(_run_verdict(args))


if __name__ == "__main__":  # pragma: no cover — точка входа CLI
    sys.exit(main())
