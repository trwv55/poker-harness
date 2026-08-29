# Poker Harness v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Телеграм-бот, который по hand history GG находит префлоп-минуса турнира (скан) и разбирает выбранную руку, с проверенным кодом расчётом — по спеке v1.0.

**Architecture:** Два процесса (`bot` + `worker`) из одного образа, общаются только через Postgres (jobs-очередь SKIP LOCKED, чекпоинты = записанные артефакты). Конвейер: парсер → нормалайзер → движок (PokerKit) → аналитическое ядро (чистый код) → изложение. LLM ровно в двух местах (vision, текст вердикта), только через фасад `harness.platform.llm` с кросс-процессным лимитером.

**Tech Stack:** Python 3.12+, uv, ruff, pyright, pydantic v2, aiogram 3, PostgreSQL 16, SQLAlchemy 2 async + Alembic, PokerKit, eval7, PydanticAI, cairosvg, structlog, pytest + testcontainers.

**Spec:** `docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md` — план аргументирует от неё; исполнитель читает оба документа.

## Global Constraints

- Python `>=3.12`; зависимости через uv; линт/формат ruff; типы pyright (strict для `src/harness/contracts`).
- Контракты — pydantic v2 с `schema_version: int`; эволюция **только опциональными полями** (старый jsonb обязан читаться новой моделью).
- **Правило зависимостей (спека §4):** пакеты `contracts`, `parsers`, `normalizer`, `engine`, `analysis`, `explanation` не импортируют ни Телеграм, ни БД, ни `harness.platform` — данные аргументами, результат значением.
- **Правило единого голоса (спека §4):** сообщения игроку собирает только `harness.presentation`; `bot`/`worker` их лишь отправляют.
- LLM зовётся **только** через `harness.platform.llm` (внедряется зависимостью); лимиты одновременности и темпа — кросс-процессные через Postgres (advisory-локи + оконный счётчик по `llm_calls`), asyncio-семафор — лишь локальный предфильтр (спека §7).
- `jobs.type` и `jobs.session_id NOT NULL` — с первой миграции (спека §6).
- Чекпоинт = записанный артефакт станции; ретрай и эскалация продолжают, не начинают заново; отправка в ТГ идемпотентна по message_id в `jobs.payload` (спека §8.2).
- Миграции — отдельный одноразовый шаг `docker compose run --rm migrate`; entrypoint'ы сервисов миграций не катят (спека §12).
- Тесты очереди, репозиториев и лимитера — на настоящем Postgres (testcontainers); без инфраструктуры — только контракты и чистый конвейер (спека §10).
- Квота — скользящее окно 24 ч SQL-счётчиком по `jobs`; ни поля пояса, ни cron-сброса (спека §9).
- Коммиты частые, каждая задача заканчивается зелёными `ruff check . && pyright && pytest`.

**Отступление от спеки, фиксируемое здесь:** §4 спеки рисует пакеты верхнего уровня (`src/platform/` и др.). Пакет `platform` затеняет одноимённый модуль stdlib для всего интерпретатора — реальный баг. Поэтому корневой пакет один: `src/harness/`, сервисные пакеты — его подпакеты (`harness.platform` stdlib не трогает). Задача 1 правит §4 спеки и коммитит правку.

---

## Карта файлов

```
pyproject.toml, uv.lock, .github/workflows/ci.yml, Dockerfile, docker-compose.yml, alembic.ini
src/harness/
  contracts/{__init__,raw,canonical,enriched,ranges,analysis}.py
  parsers/{__init__,hh_parser}.py        # vision_adapter.py — задача 22
  normalizer/{__init__,normalize}.py
  engine/{__init__,replay,validation}.py
  analysis/
    tools/{__init__,equity,pot_odds,icm,pushfold}.py
    {__init__,classifier,error_cost,preflop,scan}.py
  explanation/{__init__,verdict_text,range_render}.py   # задача 21
  presentation/{__init__,messages,keyboards}.py
  memory/{__init__,models,repos}.py
  platform/{__init__,config,llm,limiter,queue,trace}.py # eval_runner — задачи 21–22
  bot/{__init__,main,router,handlers}.py
  worker/{__init__,main,pipeline}.py
migrations/versions/0001_initial.py
tests/{conftest.py, test_contracts.py, test_regression_grid.py, test_hh_parser.py,
       test_normalizer.py, test_engine.py, test_equity.py, test_pot_odds.py, test_icm.py,
       test_pushfold.py, test_preflop_analysis.py, test_scan.py, test_memory.py,
       test_queue.py, test_llm_facade.py, test_presentation.py, test_worker_pipeline.py,
       test_bot_handlers.py}
fixtures/hh/daily-classic-146.txt, fixtures/hh/pko-bounty-172.txt
scripts/spike_vision.py                  # выкидной
evals/{vision,verdict,e2e}/              # наполняются задачами 21–22
```

Этапы: A (задачи 1–3) фундамент · B (4–8) конвейер HH · C (9–13) ядро · D (14–18) платформа · E (19–20) бот+деплой · F (21–23) дальние, крупнее — с плановыми точками детализации.

---

### Task 1: Каркас репозитория и тулинг

**Files:**
- Create: `pyproject.toml`, `.github/workflows/ci.yml`, `src/harness/__init__.py` (+ пустые `__init__.py` всех подпакетов из карты, кроме файлов задач 21–22), `tests/conftest.py`, `tests/test_scaffold.py`, `fixtures/hh/` (перенос двух GG-файлов)
- Modify: `docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md` (§4: дерево `src/harness/`, причина — stdlib-конфликт `platform`)

**Interfaces:**
- Produces: `FIXTURE_DAILY`, `FIXTURE_PKO` (pathlib.Path) в `tests/conftest.py`; рабочие `uv run pytest|ruff|pyright`.

- [ ] **Step 1: pyproject + окружение**

```toml
[project]
name = "poker-harness"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.7"]

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "ruff>=0.6", "pyright>=1.1.380"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/harness"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.12"
```

Run: `uv python install 3.12 && uv sync`

- [ ] **Step 2: пакеты и фикстуры**

Создать `src/harness/__init__.py` и пустые `__init__.py` по карте файлов. Перенести фикстуры:

```bash
mkdir -p fixtures/hh
git mv ".claude/RawHands/GG20260820-1641 - Daily Classic 4.txt" fixtures/hh/daily-classic-146.txt
git mv ".claude/RawHands/GG20260814-1930 - Bounty Hunters Deepstack Turbo 5.40.txt" fixtures/hh/pko-bounty-172.txt
```

`tests/conftest.py`:

```python
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hh"
FIXTURE_DAILY = FIXTURES / "daily-classic-146.txt"
FIXTURE_PKO = FIXTURES / "pko-bounty-172.txt"
```

- [ ] **Step 3: smoke-тест**

`tests/test_scaffold.py`:

```python
import harness
from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO

def test_package_importable():
    assert harness is not None

def test_fixtures_present():
    assert FIXTURE_DAILY.stat().st_size > 100_000
    assert FIXTURE_PKO.stat().st_size > 100_000

def test_stdlib_platform_not_shadowed():
    import platform
    assert "harness" not in (platform.__file__ or "")
```

Run: `uv run pytest -q` → 3 passed. Run: `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 4: CI**

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pyright
      - run: uv run pytest -q
```

- [ ] **Step 5: правка §4 спеки** — заменить в дереве `src/<пакеты>` на `src/harness/<подпакеты>`, добавить абзац: «Корневой пакет `harness`: пакет верхнего уровня `platform` затенял бы stdlib-модуль `platform`».

- [ ] **Step 6: Commit** — `git add -A && git commit -m "chore: каркас репо (uv, ruff, pyright, CI), фикстуры, правка §4 спеки"`

---

### Task 2: Контракты данных

**Files:**
- Create: `src/harness/contracts/raw.py`, `canonical.py`, `enriched.py`, `ranges.py`, `analysis.py`, `__init__.py` (реэкспорт)
- Test: `tests/test_contracts.py`

**Interfaces (Produces — используется всеми последующими задачами):**

```python
# raw.py
class Provenance(StrEnum): HAND_HISTORY = "hand_history"; SCREENSHOT = "screenshot"
class Street(StrEnum): PREFLOP = "preflop"; FLOP = "flop"; TURN = "turn"; RIVER = "river"
class ActionKind(StrEnum): FOLD = "fold"; CHECK = "check"; CALL = "call"; BET = "bet"; RAISE = "raise"
class PostKind(StrEnum): ANTE = "ante"; SMALL_BLIND = "small_blind"; BIG_BLIND = "big_blind"
class SeatInfo(BaseModel): seat: int; label: str; stack: int
class Post(BaseModel): label: str; kind: PostKind; amount: int
class RawAction(BaseModel):
    street: Street; label: str; kind: ActionKind
    amount: int | None = None      # calls/bets N — ДОПЛАТА, как в источнике
    to_amount: int | None = None   # raises X to Y -> Y
    is_all_in: bool = False; raw_line: str
class Uncalled(BaseModel): label: str; amount: int
class ShowdownEntry(BaseModel): label: str; cards: list[str]
class Collected(BaseModel): label: str; amount: int   # мейн/сайд не подписаны — как в GG
class SummaryInfo(BaseModel):
    total_pot: int; rake: int; jackpot: int; bingo: int; fortune: int; tax: int
    board: list[str] = []; seat_lines: list[str] = []
class VisionMeta(BaseModel):       # только для скринов
    confidence: dict[str, float] = {}; needs_review: list[str] = []
    image_hash: str | None = None; nicknames: dict[str, str] = {}
    bounties: dict[str, int] = {}; displayed_pot: int | None = None
class RawHand(BaseModel):
    schema_version: int = 1; provenance: Provenance; source_ref: str
    hand_no: str; tournament_id: str; tournament_name: str; level: int
    sb: int; bb: int; ante: int; ante_type: str = "per_player"
    timestamp: datetime; table_name: str; max_seats: int; button_seat: int
    seats: list[SeatInfo]; posts: list[Post]
    dealt: dict[str, list[str]] = {}          # пустой список = Dealt to без карт
    actions: list[RawAction] = []; boards: dict[Street, list[str]] = {}
    uncalled: list[Uncalled] = []; showdowns: list[ShowdownEntry] = []
    collected: list[Collected] = []; summary: SummaryInfo | None = None
    vision: VisionMeta | None = None
    unknown_lines: list[str] = []             # всё, что парсер не распознал

# canonical.py
class Identity(StrEnum): HERO = "hero"; NICK = "nick"; ANON = "anon"
class PlayerState(BaseModel):
    seat: int; label: str; identity: Identity; position: str  # "BTN"/"SB"/"BB"/"UTG"/...
    stack: int; stack_bb: float
class CanonicalAction(BaseModel):
    street: Street; label: str; kind: ActionKind
    committed_after: int   # ИТОГО поставлено игроком на этой улице после действия
    is_all_in: bool = False; raw_line: str
class CanonicalHand(BaseModel):
    schema_version: int = 1; provenance: Provenance
    tournament_id: str; hand_no: str; hand_index: int | None = None
    level: int; sb: int; bb: int; ante: int; ante_type: str = "per_player"
    timestamp: datetime; button_seat: int; hero_label: str = "Hero"
    players: list[PlayerState]; dealt: dict[str, list[str]] = {}
    actions: list[CanonicalAction] = []; boards: dict[Street, list[str]] = {}
    uncalled: list[Uncalled] = []; showdowns: list[ShowdownEntry] = []
    collected: list[Collected] = []; summary: SummaryInfo | None = None
    bounties: dict[str, int] | None = None; bounty_source: str | None = None
    vision: VisionMeta | None = None

# enriched.py
class SidePot(BaseModel): amount: int; eligible: list[str]
class DecisionPoint(BaseModel):
    index: int; street: Street; label: str; position: str
    to_call: int; pot_before: int; eff_stack: int; eff_stack_bb: float
    spr: float | None = None; action: CanonicalAction
    live_total: int = 0    # игроков ещё в руке на момент решения, включая Hero
    live_behind: int = 0   # из них ещё не действовавших после Hero — вход правила зоны
class ValidationStatus(StrEnum): PASS = "pass"; ESCALATE = "escalate"; REJECT = "reject"
class Verdict(BaseModel):
    status: ValidationStatus
    fields: list[str] = []; questions: list[str] = []; reasons: list[str] = []
class EngineReport(BaseModel):
    pot_by_street: dict[Street, int]; final_pot: int
    side_pots: list[SidePot] = []; stacks_end: dict[str, int]
    decision_points: list[DecisionPoint]; illegal_actions: list[str] = []
class EnrichedHand(BaseModel):
    schema_version: int = 1
    hand: CanonicalHand; report: EngineReport; verdict: Verdict

# ranges.py
RANKS = "AKQJT98765432"
def class_of(card1: str, card2: str) -> str: ...  # ("Kc","3c") -> "K3s"; ("Ah","Kd") -> "AKo"; пары "QQ"
def all_classes() -> list[str]: ...               # ровно 169
class Range(BaseModel):
    weights: dict[str, float] = {}                # класс -> 0..1; отсутствие ключа = 0
    # validator: ключи из all_classes(), значения в [0,1]
    def weight(self, cls: str) -> float: ...
    def fraction_of_hands(self) -> float: ...     # взвешенная доля комбо (пары 6, s 4, o 12; всего 1326)

# analysis.py
class Zone(StrEnum): STRICT = "strict"; ASSUMING = "assuming"
class SpotKind(StrEnum):
    PUSHFOLD_UNOPENED = "pushfold_unopened"; PUSHFOLD_FACING_SHOVE = "pushfold_facing_shove"
    PREFLOP_OTHER = "preflop_other"; POSTFLOP = "postflop"
class Assumption(BaseModel): range: Range; source: str; note: str = ""
class PointVerdict(BaseModel):
    dp_index: int; street: Street; spot: SpotKind; zone: Zone
    action_taken: str; best_action: str; ev_diff_bb: float   # <0 = потеря
    assumption: Assumption | None = None; tools: list[str] = []; detail: dict = {}
class AnalysisResult(BaseModel):
    schema_version: int = 1; hand_no: str
    points: list[PointVerdict]; ranked: list[int] = []; total_ev_loss_bb: float = 0.0
```

- [ ] **Step 1: тесты** — `tests/test_contracts.py`:

```python
import json
from harness.contracts import RawHand, Range, class_of, all_classes

def make_min_raw(**over):
    base = dict(provenance="hand_history", source_ref="f.txt", hand_no="TM1",
                tournament_id="306148954", tournament_name="Daily Classic $4", level=23,
                sb=3000, bb=6000, ante=750, timestamp="2026-08-20T22:22:36",
                table_name="8", max_seats=8, button_seat=3,
                seats=[{"seat": 4, "label": "Hero", "stack": 3891}], posts=[])
    base.update(over); return base

def test_raw_roundtrip():
    h = RawHand.model_validate(make_min_raw())
    assert RawHand.model_validate_json(h.model_dump_json()) == h

def test_old_json_readable_by_new_model():
    d = make_min_raw(); d.pop("ante_type")          # «старый» документ без нового поля
    assert RawHand.model_validate(d).ante_type == "per_player"

def test_class_of():
    assert class_of("Kc", "3c") == "K3s"
    assert class_of("3c", "Kc") == "K3s"            # порядок карт не важен
    assert class_of("Ah", "Kd") == "AKo"
    assert class_of("Qs", "Qh") == "QQ"

def test_169_classes():
    cs = all_classes()
    assert len(cs) == 169 and len(set(cs)) == 169
    assert {"AA", "AKs", "AKo", "32o"} <= set(cs)

def test_range_validates():
    r = Range(weights={"AA": 1.0, "AKs": 0.5})
    assert r.weight("AA") == 1.0 and r.weight("72o") == 0.0
    import pytest
    with pytest.raises(Exception): Range(weights={"XX": 1.0})
    with pytest.raises(Exception): Range(weights={"AA": 1.5})

def test_fraction_of_hands():
    assert abs(Range(weights={c: 1.0 for c in all_classes()}).fraction_of_hands() - 1.0) < 1e-9
    assert abs(Range(weights={"AA": 1.0}).fraction_of_hands() - 6 / 1326) < 1e-9
```

- [ ] **Step 2: убедиться, что падают** — `uv run pytest tests/test_contracts.py -q` → ImportError.
- [ ] **Step 3: реализация** — модели из Interfaces дословно; `class_of`: отсортировать по индексу в `RANKS`, пара/`s`/`o`; `fraction_of_hands`: комбо-веса 6/4/12, делить на 1326.
- [ ] **Step 4: зелёные** — `uv run pytest tests/test_contracts.py -q && uv run pyright`.
- [ ] **Step 5: Commit** — `feat: контракты RawHand/CanonicalHand/EnrichedHand/Range/AnalysisResult`

---

### Task 3: Красная регрессионная сетка (318 рук)

**Files:**
- Create: `tests/test_regression_grid.py`

**Interfaces:**
- Consumes: `harness.parsers.hh_parser.parse_file(text, source_ref) -> list[RawHand]` (задача 4), `harness.normalizer.normalize(raw) -> CanonicalHand` (задача 5), `harness.engine.enrich(hand) -> EnrichedHand` (задача 6). Тест пишется ДО них — в этом смысл «от проверки» (спека §13, EVALS этаж 1).

- [ ] **Step 1: написать сетку целиком, пометить xfail**

```python
import pytest
from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO

pytestmark = pytest.mark.xfail(reason="конвейер ещё не реализован", strict=False)  # снимается в задаче 7

def _pipeline(path):
    from harness.engine import enrich
    from harness.normalizer import normalize
    from harness.parsers.hh_parser import parse_file
    raws = parse_file(path.read_text(encoding="utf-8"), source_ref=path.name)
    return raws, [enrich(normalize(r)) for r in raws]

@pytest.mark.parametrize("path,expected_hands", [(FIXTURE_DAILY, 146), (FIXTURE_PKO, 172)])
def test_grid(path, expected_hands):
    raws, enriched = _pipeline(path)
    assert len(raws) == expected_hands
    # руки отсортированы хронологически (в файле GG — обратный порядок)
    ts = [r.timestamp for r in raws]
    assert ts == sorted(ts)
    # формат покрыт полностью: ни одной нераспознанной строки
    assert all(r.unknown_lines == [] for r in raws), \
        [line for r in raws for line in r.unknown_lines][:5]
    for en in enriched:
        # каждая рука проходит валидатор без эскалаций (HH = факт)
        assert en.verdict.status == "pass", (en.hand.hand_no, en.verdict)
        # банк, пересчитанный движком, сходится с SUMMARY (rake в фикстурах = 0)
        assert en.report.final_pot == en.hand.summary.total_pot, en.hand.hand_no
        assert en.hand.summary.rake == 0
        # стеки на конец руки неотрицательны, сумма стеков сохранилась
        assert all(v >= 0 for v in en.report.stacks_end.values())
        start = sum(p.stack for p in en.hand.players)
        end = sum(en.report.stacks_end.values())
        assert start == end + 0  # rake 0: фишки не исчезают
```

- [ ] **Step 2: убедиться, что xfail** — `uv run pytest tests/test_regression_grid.py -q` → 2 xfailed.
- [ ] **Step 3: Commit** — `test: красная регрессионная сетка на 318 реальных рук (xfail до задачи 7)`

---

### Task 4: HHParser — формат GG

**Files:**
- Create: `src/harness/parsers/hh_parser.py`
- Test: `tests/test_hh_parser.py`

**Interfaces:**
- Produces: `parse_file(text: str, source_ref: str) -> list[RawHand]` — руки в хронологическом порядке; `parse_hand(block: str, source_ref: str) -> RawHand`.

- [ ] **Step 1: тест на реальной руке (вербатим из фикстуры)**

```python
from harness.contracts import ActionKind, PostKind, Street
from harness.parsers.hh_parser import parse_file, parse_hand
from tests.conftest import FIXTURE_DAILY

SAMPLE = """Poker Hand #TM6316081388: Tournament #306148954, Daily Classic $4 Hold'em No Limit - Level23(3,000/6,000(750)) - 2026/08/20 22:22:36
Table '8' 8-max Seat #3 is the button
Seat 1: c30a7c9e (85,440 in chips)
Seat 2: bb4aa4e0 (25,109 in chips)
Seat 3: c3986130 (222,896 in chips)
Seat 4: Hero (3,891 in chips)
Seat 5: fcc9bf19 (415,055 in chips)
Seat 7: 5553a2cd (70,471 in chips)
Seat 8: 95b4992 (151,005 in chips)
95b4992: posts the ante 750
c3986130: posts the ante 750
fcc9bf19: posts the ante 750
bb4aa4e0: posts the ante 750
5553a2cd: posts the ante 750
c30a7c9e: posts the ante 750
Hero: posts the ante 750
Hero: posts small blind 3,000
fcc9bf19: posts big blind 6,000
*** HOLE CARDS ***
Dealt to c30a7c9e 
Dealt to bb4aa4e0 
Dealt to c3986130 
Dealt to Hero [3c Kc]
Dealt to fcc9bf19 
Dealt to 5553a2cd 
Dealt to 95b4992 
5553a2cd: raises 63,000 to 69,000
95b4992: folds
c30a7c9e: folds
bb4aa4e0: folds
c3986130: folds
Hero: calls 141 and is all-in
fcc9bf19: folds
Uncalled bet (63,000) returned to 5553a2cd
5553a2cd: shows [Js Ah]
Hero: shows [3c Kc]
*** FLOP *** [Kd Td 3s]
*** TURN *** [Kd Td 3s] [Qd]
*** RIVER *** [Kd Td 3s Qd] [2d]
*** SHOWDOWN ***
5553a2cd collected 14,673 from pot
5553a2cd collected 5,718 from pot
*** SUMMARY ***
Total pot 20,391 | Rake 0 | Jackpot 0 | Bingo 0 | Fortune 0 | Tax 0
Board [Kd Td 3s Qd 2d]
Seat 1: c30a7c9e folded before Flop
Seat 2: bb4aa4e0 folded before Flop
Seat 3: c3986130 (button) folded before Flop
Seat 4: Hero (small blind) showed [3c Kc] and lost with two pair, Kings and Threes
Seat 5: fcc9bf19 (big blind) folded before Flop
Seat 7: 5553a2cd showed [Js Ah] and won (20,391) with a straight, Ace to Ten
Seat 8: 95b4992 folded before Flop
"""

def test_parse_sample_hand():
    h = parse_hand(SAMPLE, source_ref="daily-classic-146.txt")
    assert h.hand_no == "TM6316081388" and h.tournament_id == "306148954"
    assert (h.level, h.sb, h.bb, h.ante) == (23, 3000, 6000, 750)
    assert h.max_seats == 8 and h.button_seat == 3
    assert len(h.seats) == 7 and h.seats[3].label == "Hero" and h.seats[3].stack == 3891
    assert sum(1 for p in h.posts if p.kind == PostKind.ANTE) == 7
    assert h.dealt["Hero"] == ["3c", "Kc"] and h.dealt["c30a7c9e"] == []
    raise_a = h.actions[0]
    assert (raise_a.kind, raise_a.amount, raise_a.to_amount) == (ActionKind.RAISE, 63000, 69000)
    call_a = next(a for a in h.actions if a.label == "Hero")
    assert (call_a.kind, call_a.amount, call_a.is_all_in) == (ActionKind.CALL, 141, True)
    assert h.boards[Street.FLOP] == ["Kd", "Td", "3s"] and h.boards[Street.RIVER] == ["2d"]
    assert [c.amount for c in h.collected] == [14673, 5718]
    assert h.uncalled[0].amount == 63000
    assert h.summary.total_pot == 20391 and h.summary.tax == 0
    assert h.showdowns[0].cards == ["Js", "Ah"]
    assert h.unknown_lines == []

def test_parse_file_sorted_and_counted():
    raws = parse_file(FIXTURE_DAILY.read_text(encoding="utf-8"), source_ref="daily")
    assert len(raws) == 146
    ts = [r.timestamp for r in raws]
    assert ts == sorted(ts)          # файл GG — в обратном порядке, парсер сортирует
```

- [ ] **Step 2: убедиться, что падает** — ImportError/AssertionError.
- [ ] **Step 3: реализация** — построчный конечный автомат со словарём регэкспов; суммы `[\d,]+` → `int(x.replace(",", ""))`. Ключевые паттерны:

```python
RE_HEADER = re.compile(r"^Poker Hand #(?P<no>\S+): Tournament #(?P<tid>\d+), (?P<name>.+) Hold'em No Limit - Level(?P<lvl>\d+)\((?P<sb>[\d,]+)/(?P<bb>[\d,]+)\((?P<ante>[\d,]+)\)\) - (?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})$")
RE_TABLE = re.compile(r"^Table '(?P<t>[^']+)' (?P<mx>\d+)-max Seat #(?P<btn>\d+) is the button$")
RE_SEAT = re.compile(r"^Seat (?P<s>\d+): (?P<l>\S+) \((?P<st>[\d,]+) in chips\)$")
RE_ANTE = re.compile(r"^(?P<l>\S+): posts the ante (?P<a>[\d,]+)$")
RE_BLIND = re.compile(r"^(?P<l>\S+): posts (?P<k>small|big) blind (?P<a>[\d,]+)$")
RE_DEALT = re.compile(r"^Dealt to (?P<l>\S+)(?: \[(?P<c>[^\]]+)\])?\s*$")
RE_ACTION = re.compile(r"^(?P<l>\S+): (?P<v>folds|checks|calls (?P<ca>[\d,]+)|bets (?P<ba>[\d,]+)|raises (?P<ra>[\d,]+) to (?P<rt>[\d,]+))(?P<ai> and is all-in)?$")
RE_UNCALLED = re.compile(r"^Uncalled bet \((?P<a>[\d,]+)\) returned to (?P<l>\S+)$")
RE_SHOWS = re.compile(r"^(?P<l>\S+): shows \[(?P<c>[^\]]+)\]")
RE_COLLECTED = re.compile(r"^(?P<l>\S+) collected (?P<a>[\d,]+) from pot$")
RE_STREET = re.compile(r"^\*\*\* (?P<s>HOLE CARDS|FLOP|TURN|RIVER|SHOWDOWN|SUMMARY) \*\*\*(?: \[(?P<b>[^\]]+)\])?(?: \[(?P<b2>[^\]]+)\])?$")
RE_TOTAL = re.compile(r"^Total pot (?P<tp>[\d,]+) \| Rake (?P<r>[\d,]+) \| Jackpot (?P<j>[\d,]+) \| Bingo (?P<bi>[\d,]+) \| Fortune (?P<f>[\d,]+) \| Tax (?P<tx>[\d,]+)$")
RE_BOARD = re.compile(r"^Board \[(?P<b>[^\]]+)\]$")
```

Правила: TURN/RIVER — доска из последней скобки (`b2` при наличии, иначе `b`); строки `Seat N: … folded/showed/won …` после SUMMARY → `summary.seat_lines`; пустые строки — разделители; всё нераспознанное → `unknown_lines` (сетка задачи 7 заставит закрыть весь формат, включая отличия PKO-файла); `parse_file` делит по `^Poker Hand #`, сортирует по `timestamp` (при равенстве — по `hand_no`).

- [ ] **Step 4: зелёные** — `uv run pytest tests/test_hh_parser.py -q`.
- [ ] **Step 5: Commit** — `feat: HH-парсер формата GG (реальные фикстуры, unknown_lines для полноты)`

---

### Task 5: Normalizer

**Files:**
- Create: `src/harness/normalizer/normalize.py` (+реэкспорт в `__init__.py`)
- Test: `tests/test_normalizer.py`

**Interfaces:**
- Produces: `normalize(raw: RawHand) -> CanonicalHand`; `POSITIONS_BY_COUNT: dict[int, list[str]]` — позиции в порядке мест от SB.

- [ ] **Step 1: тесты**

```python
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from tests.test_hh_parser import SAMPLE

def test_positions_and_bb():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    pos = {p.label: p.position for p in h.players}
    assert pos["Hero"] == "SB" and pos["fcc9bf19"] == "BB"
    assert pos["c3986130"] == "BTN" and pos["5553a2cd"] == "UTG"
    hero = next(p for p in h.players if p.label == "Hero")
    assert hero.identity == "hero" and abs(hero.stack_bb - 3891 / 6000) < 1e-9
    anon = next(p for p in h.players if p.label == "c30a7c9e")
    assert anon.identity == "anon"

def test_committed_after_unified():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    acts = {(a.label, a.kind): a for a in h.actions}
    assert acts[("5553a2cd", "raise")].committed_after == 69000
    # Hero: SB 3000 + call 141 = 3144? нет: блайнд 3000 + доплата 141 = 3141
    assert acts[("Hero", "call")].committed_after == 3141
    assert acts[("Hero", "call")].is_all_in

def test_heads_up_button_is_sb():
    # синтетика: 2 игрока, кнопка = SB
    raw = parse_hand(SAMPLE, source_ref="x").model_copy(deep=True)
    raw.seats = raw.seats[:2]  # места 1 и 2
    raw.button_seat = raw.seats[0].seat
    raw.posts = [p for p in raw.posts if p.label in {raw.seats[0].label, raw.seats[1].label}]
    raw.actions, raw.dealt, raw.showdowns, raw.collected = [], {}, [], []
    h = normalize(raw)
    assert [p.position for p in h.players] == ["BTN", "BB"]  # HU: кнопка ставит SB
```

- [ ] **Step 2: убедиться, что падают.**
- [ ] **Step 3: реализация**

```python
POSITIONS_BY_COUNT = {
    2: ["BTN", "BB"],                       # HU: BTN = SB
    3: ["SB", "BB", "BTN"],
    4: ["SB", "BB", "UTG", "BTN"],
    5: ["SB", "BB", "UTG", "CO", "BTN"],
    6: ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
    7: ["SB", "BB", "UTG", "UTG+1", "HJ", "CO", "BTN"],
    8: ["SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"],
    9: ["SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"],
}
```

Порядок мест: занятые места по кругу начиная со следующего после кнопки (для HU — с кнопки). `committed_after` — аккумулятор по (label, street): блайнды входят в префлоп-коммит; `call N`/`bets N` — прибавка; `raises X to Y` — установка в Y; анте в коммит улицы НЕ входит (он в банке отдельно). Identity: `Hero` → hero; из `vision.nicknames` → nick; иначе anon. `hand_index` проставляет вызывающий (parse_file порядковый номер после сортировки — добавить проставление в `parse_file` или тут оставить None; выбрано: None, проставляет воркер при сохранении).

- [ ] **Step 4: зелёные + pyright.**
- [ ] **Step 5: Commit** — `feat: нормалайзер (позиции, bb, унификация коммитов, identity)`

---

### Task 6: Движок — replay (PokerKit) + validation

**Files:**
- Create: `src/harness/engine/replay.py`, `src/harness/engine/validation.py`, `src/harness/engine/__init__.py` (`enrich`)
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `CanonicalHand`.
- Produces: `replay(hand: CanonicalHand) -> EngineReport`; `validate(hand: CanonicalHand, report: EngineReport) -> Verdict`; `enrich(hand: CanonicalHand) -> EnrichedHand` (replay + validate, единственный прогон — спека §3 арх.).
- Зависимость: `uv add "pokerkit>=0.7,<0.8"`. Точные сигнатуры PokerKit сверять с документацией пина; поведение пришпилено тестами на суммы.

- [ ] **Step 1: тесты**

```python
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from tests.test_hh_parser import SAMPLE

def test_sample_hand_pot_and_stacks():
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    assert en.verdict.status == "pass"
    assert en.report.final_pot == 20391                      # сходится с SUMMARY
    # виллан: анте 750 + нетто-вклад 6000 (рейз до 69000, 63000 возвращено), забрал банк
    assert en.report.stacks_end["5553a2cd"] == 70471 - 6750 + 20391  # = 84112
    assert en.report.stacks_end["Hero"] == 0                 # проиграл олин
    # сохранение фишек
    start = 85440 + 25109 + 222896 + 3891 + 415055 + 70471 + 151005
    assert sum(en.report.stacks_end.values()) == start

def test_decision_point_for_hero():
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    assert dp.street == "preflop"
    assert dp.to_call == 141                                  # доплата при стеке 141 за анте+SB
    assert dp.action.kind == "call" and dp.action.is_all_in
    assert (dp.live_total, dp.live_behind) == (3, 1)          # рейзер, Hero, BB; после Hero — BB

def test_validation_rejects_chip_mismatch():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    bad = h.model_copy(deep=True)
    bad.summary.total_pot = 99999                            # банк не сходится
    en = enrich(bad)
    assert en.verdict.status == "reject"                     # HH = факт -> баг парсера, не эскалация
    assert any("pot" in r for r in en.verdict.reasons)

def test_validation_rejects_duplicate_cards():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    bad = h.model_copy(deep=True)
    bad.dealt["Hero"] = ["Js", "Ah"]                         # дубль карт вскрытия оппонента
    en = enrich(bad)
    assert en.verdict.status == "reject"

def test_validation_escalates_for_screenshot():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    scr = h.model_copy(deep=True)
    scr.provenance = "screenshot"
    scr.summary.total_pot = 99999
    en = enrich(scr)
    assert en.verdict.status == "escalate"                   # скрин = гипотеза -> спросить игрока
    assert en.verdict.fields                                  # какие поля переспрашивать
```

- [ ] **Step 2: убедиться, что падают.**
- [ ] **Step 3: реализация `replay`** — обвязка PokerKit:

```python
from pokerkit import Automation, NoLimitTexasHoldem
# порядок игроков для pokerkit = порядок players из нормалайзера (от SB; HU — от BTN)
state = NoLimitTexasHoldem.create_state(
    automations=(Automation.ANTE_POSTING, Automation.BET_COLLECTION,
                 Automation.BLIND_OR_STRADDLE_POSTING, Automation.CARD_BURNING,
                 Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
                 Automation.HAND_KILLING, Automation.RUNOUT_COUNT_SELECTION),
    ante_trimming_status=False,          # GG: анте per-player
    raw_antes=hand.ante,
    raw_blinds_or_straddles=(hand.sb, hand.bb),
    min_bet=hand.bb,
    raw_starting_stacks=[p.stack for p in hand.players],
    player_count=len(hand.players),
)
```

Карты: `state.deal_hole(...)` по `dealt` (неизвестные — из остатка колоды детерминированно, они не влияют на банк); действия по улицам: fold → `state.fold()`, check/call → `state.check_or_call()`, bet/raise → `state.complete_bet_or_raise_to(committed_after)`; доски — `state.deal_board(...)`. Перед каждым действием Hero-игрока снять `DecisionPoint` (to_call = `state.checking_or_calling_amount`, pot_before = `state.total_pot_amount`, eff_stack = min(стек героя, макс. стек живого оппонента), spr = eff/pot на постфлопе, `live_total` = не сфолдившие на этот момент, `live_behind` = из них те, чей ход после Hero в текущем круге). Несовпадение ожидаемого актёра/нелегальное действие → записать в `illegal_actions`, прервать реплей. `pot_by_street` фиксировать на границах улиц; `final_pot` = `state.total_pot_amount` до раздачи выигрышей; `stacks_end` = `state.stacks` после.

- [ ] **Step 4: реализация `validate`** — политика (движок не судит — спека §3 арх.):

```python
def validate(hand, report) -> Verdict:
    reasons, fields = [], []
    if report.illegal_actions: reasons.append(f"illegal: {report.illegal_actions}")
    if hand.summary and report.final_pot != hand.summary.total_pot:
        reasons.append(f"pot mismatch: engine={report.final_pot} summary={hand.summary.total_pot}")
        fields += ["stacks", "actions"]
    if _has_duplicate_cards(hand): reasons.append("duplicate cards"); fields.append("cards")
    start = sum(p.stack for p in hand.players)
    if sum(report.stacks_end.values()) + (hand.summary.rake if hand.summary else 0) != start:
        reasons.append("chip conservation violated")
    if not reasons: return Verdict(status="pass")
    if hand.provenance == Provenance.SCREENSHOT:
        return Verdict(status="escalate", fields=sorted(set(fields)),
                       questions=[_question_for(f, hand) for f in sorted(set(fields))],
                       reasons=reasons)
    return Verdict(status="reject", reasons=reasons)   # HH = факт: чинить парсер, не данные
```

`_question_for` — короткий русский вопрос по полю («Стек героя …?»). Валидатор **никогда не правит данные**.

- [ ] **Step 5: зелёные** — `uv run pytest tests/test_engine.py -q`.
- [ ] **Step 6: Commit** — `feat: движок руки — PokerKit-реплей + валидатор (pass/escalate/reject)`

---

### Task 7: Зелёная сетка на 318 руках

**Files:**
- Modify: `tests/test_regression_grid.py` (снять `pytestmark = xfail`), `src/harness/parsers/hh_parser.py`, `src/harness/engine/*` (по находкам)

- [ ] **Step 1:** удалить строку `pytestmark = pytest.mark.xfail(...)`.
- [ ] **Step 2:** `uv run pytest tests/test_regression_grid.py -q -x` — гонять и чинить до зелени. Ожидаемые находки: не покрытые паттерны PKO-файла (уйдут из `unknown_lines`), сплит-поты/нечётные фишки, ранние уходы игроков. Валидатор и движок не ослаблять под тест — чинить парсинг/реплей; если строка формата легитимно не влияет на разбор, парсер распознаёт её явно (не через unknown_lines).
- [ ] **Step 3:** полный прогон `uv run pytest -q && uv run ruff check . && uv run pyright` — всё зелёное.
- [ ] **Step 4: Commit** — `feat: регрессионная сетка зелёная на 318 реальных руках GG`

---

### Task 8 (спека §13 шаг 2a): Vision-spike — выкидной

Можно выполнять параллельно задачам 4–13; требует скриншоты от владельца продукта (5–10 штук в `scratch/screens/`, в git не коммитятся).

**Files:**
- Create: `scripts/spike_vision.py` (throwaway, помечен в шапке), `docs/superpowers/specs/2026-08-28-spike-vision-findings.md` (записка)

- [ ] **Step 1:** `uv add pydantic-ai --group dev`. Скрипт: pydantic-схема `SpikeExtraction` (стеки, карты Hero, блайнды/анте, действия, банк, ники, баунти — плоско, без RawHand), цикл по картинкам × моделям (2–3: Sonnet-класс, Haiku-класс, одна не-Anthropic — модели строками аргументов CLI), вывод JSON рядом с картинкой.
- [ ] **Step 2:** прогнать на скринах владельца, **глазами** сравнить с реальностью столов.
- [ ] **Step 3:** записка: точность по типам полей на глаз, какая модель стартовая для §7 спеки, что ломается (перекрытия? мультивей-банки? баунти?), эскиз промпта. Записка — вход задачи 22 (её плановая точка детализации).
- [ ] **Step 4: Commit** — `spike: разведка vision-извлечения (выкидной скрипт + записка выводов)`

---

### Task 9: Инструменты — эквити (eval7) и пот-оддсы

**Files:**
- Create: `src/harness/analysis/tools/equity.py`, `src/harness/analysis/tools/pot_odds.py`
- Test: `tests/test_equity.py`, `tests/test_pot_odds.py`

**Interfaces:**
- Produces: `equity_vs_range(hero: tuple[str, str], rng: Range, board: list[str] = [], *, iterations: int = 200_000, seed: int = 42) -> float`; `equity_vs_ranges(hero: tuple[str, str], ranges: Sequence[Range], board: list[str] = [], *, iterations: int = 100_000, seed: int = 42) -> float` — доля банка героя при вскрытии против нескольких диапазонов сразу (ничьи делятся поровну; карточные коллизии между сэмплами оппонентов отбрасываются) — нужна мультивей-шову задачи 11; `equity_hand_vs_hand(h1, h2, board=[]) -> float`; `required_equity(to_call: int, pot_before: int) -> float`.
- Зависимость: `uv add eval7`. **Contingency:** если у eval7 нет колеса под Python 3.12 и сборка из исходников не взлетает за ~час — реализовать `equity.py` на PokerKit-эвалуаторе с собственным MC-сэмплером (интерфейс тот же, тесты те же, скорость проверить: ≥50k итераций/с достаточно), решение зафиксировать строкой в спеке §3.

- [ ] **Step 1: тесты (якоря из EVALS.md, этаж 1)**

```python
from harness.analysis.tools.equity import equity_hand_vs_hand, equity_vs_range
from harness.analysis.tools.pot_odds import required_equity
from harness.contracts import Range

def test_anchor_aks_vs_qq():
    assert abs(equity_hand_vs_hand(("As", "Ks"), ("Qh", "Qd")) - 0.46) < 0.015

def test_anchor_aa_vs_kk():
    assert abs(equity_hand_vs_hand(("Ah", "Ad"), ("Kh", "Kd")) - 0.815) < 0.02

def test_equity_vs_range_monotone():
    wide = Range(weights={c: 1.0 for c in ["22", "33", "A2s", "K9o", "QTs", "76s"]})
    tight = Range(weights={"AA": 1.0, "KK": 1.0})
    e_wide = equity_vs_range(("Qs", "Qh"), wide)
    e_tight = equity_vs_range(("Qs", "Qh"), tight)
    assert e_wide > 0.6 > e_tight        # QQ впереди широкого, позади AA/KK
    assert e_tight < 0.25

def test_equity_deterministic_with_seed():
    r = Range(weights={"AKo": 1.0, "AKs": 1.0})
    assert equity_vs_range(("Th", "Td"), r) == equity_vs_range(("Th", "Td"), r)

def test_multiway_equity_below_headsup():
    from harness.analysis.tools.equity import equity_vs_ranges
    r = Range(weights={"AKo": 1.0, "AKs": 1.0})
    hu = equity_vs_ranges(("Qs", "Qh"), [r])
    three = equity_vs_ranges(("Qs", "Qh"), [r, r])
    assert hu > three            # против двух AK доля банка меньше, чем против одного
    assert 0.35 < three < 0.55   # QQ против двух AK — примерно паритет

def test_required_equity():
    assert abs(required_equity(50, 100) - 50 / 150) < 1e-9   # пот 100 (ставка внутри), колл 50
    assert abs(required_equity(141, 20250) - 141 / 20391) < 1e-9  # рука из фикстуры
```

- [ ] **Step 2: падают.**
- [ ] **Step 3: реализация** — маппинг `Range` → взвешенный список eval7-комбо (раскрытие класса в конкретные комбо с фильтром против карт hero/board), MC с `eval7.py_hand_vs_range_monte_carlo` либо ручной сэмплер по весам + `eval7.evaluate`; сид фиксируется. `required_equity(to_call, pot_before) = to_call / (pot_before + to_call)` — `pot_before` уже содержит ставку соперника.
- [ ] **Step 4: зелёные.** Step 5: Commit — `feat: эквити против взвешенного диапазона + пот-оддсы (якорные тесты)`

---

### Task 10: Инструменты — ICM (Малмут-Харвилл)

**Files:**
- Create: `src/harness/analysis/tools/icm.py`
- Test: `tests/test_icm.py`

**Interfaces:**
- Produces: `icm_equities(stacks: list[int], payouts: list[float]) -> list[float]` — доля призового фонда каждому; `len(payouts) <= len(stacks)`, payouts по убыванию.

- [ ] **Step 1: тесты (якоря EVALS.md)**

```python
import pytest
from harness.analysis.tools.icm import icm_equities

def test_equal_stacks_equal_equity():
    eq = icm_equities([5000, 5000, 5000], [0.5, 0.3, 0.2])
    assert all(abs(e - 1 / 3) < 1e-9 for e in eq)

def test_two_players_known():
    eq = icm_equities([3000, 1000], [0.6, 0.4])
    # P(1st) = 0.75/0.25 -> 0.75*0.6+0.25*0.4 = 0.55
    assert abs(eq[0] - 0.55) < 1e-9 and abs(eq[1] - 0.45) < 1e-9

def test_sums_to_total():
    eq = icm_equities([7000, 2000, 1000], [0.5, 0.3, 0.2])
    assert abs(sum(eq) - 1.0) < 1e-9
    assert eq[0] > eq[1] > eq[2]

def test_chip_lead_worth_less_than_linear():
    eq = icm_equities([9000, 500, 500], [0.5, 0.3, 0.2])
    assert eq[0] < 0.5 * (9000 / 10000) / 0.5   # нелинейность ICM: фишки дорожают убывающе
```

- [ ] **Step 2: падают.** Step 3: реализация — рекурсия Малмута-Харвилла: P(place k) через произведения stack/оставшаяся масса, мемоизация по frozenset индексов; сложность допустима до 9 игроков × 3–4 платных места (v1: ограничить `len(payouts) <= 4`, иначе ValueError с текстом — для MTT-спотов передавать хвост призовой структуры релевантного стола).
- [ ] **Step 4: зелёные.** Step 5: Commit — `feat: ICM Малмут-Харвилл с якорными тестами`

---

### Task 11: Инструменты — пуш-фолд: EV шова/колла + HU-Нэш (fictitious play)

**Files:**
- Create: `src/harness/analysis/tools/pushfold.py`, `src/harness/analysis/tools/data/` (кэш равновесий, генерится кодом)
- Test: `tests/test_pushfold.py`

**Interfaces:**
- Produces:
  - `shove_ev_bb(hero_cls: str, hero_behind_bb: float, pot_dead_bb: float, callers: list[CallerModel], *, equity_fn=equity_vs_ranges, call_prob_fn=None) -> float` — EV шова в bb **относительно фолда** (фолд = 0; уже поставленное в банк — sunk, в базлайне потеряно). `hero_behind_bb` — стек за спиной после постов, `pot_dead_bb` — весь банк на момент решения. `call_prob_fn(caller, hero_cls) -> float`, по умолчанию — доля комбо колл-диапазона с поправкой на блокеры карт героя;
  - `CallerModel(call_range: Range, behind_bb: float)` — `behind_bb` — стек коллера за спиной;
  - `BRACKET_TIGHT`, `BRACKET_WIDE: Callable[[float], Range]` — узкая («только премиум»: AA-JJ, AK) и широкая (top-40%) модели колл-диапазона по глубине; вход bracket-теста зоны (задача 12);
  - `call_shove_ev_bb(hero_cls, hero_bb, shover_range: Range, pot_bb, to_call_bb, *, equity_fn=...) -> float`;
  - `nash_hu(eff_bb: float) -> tuple[Range, Range]` — (SB push, BB call), кэш в data/;
  - `fold_equity_ok(callers: list[CallerModel]) -> bool` — fold-equity check (все фолдят с вероятностью < 1 → шов не «бесплатный»); обязателен перед вердиктом о шове (спека, арх. §4).
- `equity_fn` внедряется — юнит-тесты EV работают на стабе без MC.

- [ ] **Step 1: тесты**

```python
from harness.analysis.tools.pushfold import CallerModel, nash_hu, shove_ev_bb
from harness.contracts import Range

def test_shove_ev_headsup_computed():
    # Hero SB: стек 10bb, поставил 0.5 -> за спиной 9.5; BB за спиной 9.0; в банке 1.5
    # фолд-ветка (0.6): +1.5bb (весь банк, включая свои 0.5 — они в базлайне уже потеряны)
    # колл-ветка (0.4): банк 1.5+9.5+9.0=20 -> 0.4*20 - 9.5 = -1.5bb
    # EV = 0.6*1.5 + 0.4*(-1.5) = +0.3bb
    caller = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0)
    ev = shove_ev_bb("32o", hero_behind_bb=9.5, pot_dead_bb=1.5, callers=[caller],
                     equity_fn=lambda *a, **k: 0.40, call_prob_fn=lambda *a: 0.4)
    assert abs(ev - 0.3) < 1e-9

def test_shove_ev_multiway_enumerates_subsets():
    # два игрока позади, каждый коллит с p=0.5; эквити героя при любом колле = 0
    # оба фолдят (0.25): +2.0bb; любая колл-ветка (0.75): -9.0bb (весь стек за спиной)
    c = CallerModel(call_range=Range(weights={"AA": 1.0}), behind_bb=9.0)
    ev = shove_ev_bb("32o", hero_behind_bb=9.0, pot_dead_bb=2.0, callers=[c, c],
                     equity_fn=lambda *a, **k: 0.0, call_prob_fn=lambda *a: 0.5)
    assert abs(ev - (0.25 * 2.0 + 0.75 * -9.0)) < 1e-9   # перебор 4 веток, не попарно

def test_nash_hu_anchors_10bb():
    push, call = nash_hu(10.0)
    assert push.weight("AA") == 1.0 and call.weight("AA") == 1.0   # AA всегда
    assert push.weight("22") > 0.9                                  # мелкие пары пушатся на 10bb
    assert push.weight("32o") < 0.1                                 # мусор — фолд на 10bb
    assert call.weight("32o") < 0.05                                # и тем более не колл

def test_nash_monotone_by_depth():
    p5, _ = nash_hu(5.0); p10, _ = nash_hu(10.0)
    assert p5.fraction_of_hands() >= p10.fraction_of_hands()        # мельче — шире

def test_nash_cached_deterministic(tmp_path):
    a, _ = nash_hu(8.0); b, _ = nash_hu(8.0)
    assert a == b
```

- [ ] **Step 2: падают.** Step 3: реализация:
  - `shove_ev_bb`: **перебор подмножеств коллеров** (не попарное приближение). Для n игроков позади перебираются все 2^n веток «кто коллит» (n ≤ 7 → ≤128 веток; ветки с вероятностью < 1e-4 отбрасываются). Вероятность ветки = Π p_call(коллеры) × Π (1 − p_call)(фолдеры). Ветка «все сфолдили» → +`pot_dead_bb`. Ветка с набором S: банк = `pot_dead_bb` + `hero_behind_bb` + Σ min(behind_i, hero_behind_bb) для i ∈ S, эквити = `equity_fn(hero_cls, [call_range_i for i in S])`, вклад ветки = эквити × банк − `hero_behind_bb`; при коллере короче героя остаток его стека в контест не входит (сайд-пот герою недоступен — учитывается через min). `call_prob_fn` по умолчанию = доля комбо `call_range` после исключения карт героя. **Арифметика точная при заданных диапазонах; приближение — только сами колл-диапазоны, и оно помечается зоной (задача 12).**
  - `nash_hu(eff_bb)`: fictitious play на матрице 169×169 — предвычислить эквити всех пар классов (симметричная таблица 169×169, MC 50k/пара с сидом, генерится один раз в `data/eq169.npy`, ~30 сек, коммитится); итерации: лучшая чистая стратегия против средней смешанной оппонента, усреднение, 200–500 итераций до сходимости |ΔEV| < 0.001bb; результат в `data/nash_hu_{eff_bb}.json`.
  - `fold_equity_ok`: False если суммарная вероятность колла ≈ 1 (все диапазоны колла покрывают всё) — тогда «шов ради фолдов» не аргумент.
- [ ] **Step 4: зелёные** (nash-тесты помечены `@pytest.mark.slow`, в CI входят). Step 5: Commit — `feat: пуш-фолд EV + HU-Нэш fictitious play + fold-equity check`

---

### Task 12: Классификатор спота, префлоп-анализ, оценщик ошибки

**Files:**
- Create: `src/harness/analysis/classifier.py`, `src/harness/analysis/preflop.py`, `src/harness/analysis/error_cost.py`, `src/harness/analysis/__init__.py` (`analyze_hand`)
- Test: `tests/test_preflop_analysis.py`

**Interfaces:**
- Consumes: `EnrichedHand`, инструменты задач 9–11.
- Produces: `analyze_hand(en: EnrichedHand) -> AnalysisResult` — v1: только префлоп-точки Hero, зона strict; `classify(dp: DecisionPoint, en: EnrichedHand) -> SpotKind`; `rank_points(points: list[PointVerdict]) -> list[int]` (индексы по убыванию потери).
- Политика v1 (из спеки §5.5, арх. «дисциплина»): судим против диапазона на момент решения, не против вскрытия; порог пуш-фолд парадигмы: eff_stack_bb ≤ 15 и (спот не открыт рейзом не-олином).
- Produces: `zone_for(best_tight: str, best_wide: str, *, live_total: int) -> tuple[Zone, str]` — правило зоны спеки §5.5: при `live_total == 2` (HU) зона `strict` («равновесие»); при 3+ живых — `strict`, если `best_tight == best_wide` (bracket стабилен, вывод от диапазонов не зависит), иначе `assuming` («модель диапазонов»).
- **Область `nash_hu` (замечание ревью):** равновесие HU применяется **только** при `live_total == 2` — настоящий HU-стол или SB-vs-BB после фолдов. При 3+ живых `nash_hu` не выдаётся за равновесие: его колл-сторона используется как *модель* колл-диапазона каждого игрока позади (по его глубине), и точка помечается зоной по `zone_for`.

- [ ] **Step 1: тесты**

```python
from harness.analysis import analyze_hand
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from tests.test_hh_parser import SAMPLE

def test_fixture_hand_correct_call_not_flagged():
    # Hero 0.52bb в SB c K3s коллит 141 в банк 20k+ — тривиально верный колл
    res = analyze_hand(enrich(normalize(parse_hand(SAMPLE, source_ref="x"))))
    hero_points = [p for p in res.points if p.spot == "pushfold_facing_shove"]
    assert len(hero_points) == 1
    p = hero_points[0]
    # strict здесь не по улице, а по правилу зоны: колл 141 в банк 20k верен против любого
    # диапазона -> bracket стабилен, допущение не несёт нагрузки
    assert p.zone == "strict" and p.assumption is None
    assert p.ev_diff_bb >= -0.05                             # не ошибка
    assert p.best_action == "call" and p.action_taken == "call"

def test_synthetic_bad_open_shove_flagged():
    # HU 10bb, Hero BTN/SB открывает олином 32o — по Нэшу фолд, EV-разница < 0
    en = _make_hu_shove_hand(hero_cards=("3c", "2d"), eff_bb=10.0)   # хелпер в этом же файле
    res = analyze_hand(en)
    p = res.points[0]
    assert p.spot == "pushfold_unopened" and p.ev_diff_bb < -0.3
    assert p.best_action == "fold" and p.action_taken == "shove"
    assert res.ranked[0] == 0 and res.total_ev_loss_bb <= p.ev_diff_bb

def test_zone_rule_direct():
    from harness.analysis.preflop import zone_for
    assert zone_for("shove", "shove", live_total=2)[0] == "strict"    # HU: равновесие
    assert zone_for("shove", "shove", live_total=5)[0] == "strict"    # bracket стабилен
    z, why = zone_for("shove", "fold", live_total=5)
    assert z == "assuming" and why                                    # вердикт зависит от модели

def test_multiway_shove_zone_invariant():
    en = _make_multiway_shove_hand(hero_cards=("Ac", "Ts"), eff_bb=12.0, players_behind=3)
    for p in analyze_hand(en).points:
        assert (p.assumption is not None) == (p.zone == "assuming")   # допущение показано всегда
        if p.zone == "assuming":
            assert "мультивей" in p.assumption.note or "модел" in p.assumption.note

def test_range_independent_call_is_strict():
    # колл шова с AA верен против любого диапазона -> bracket стабилен -> strict
    en = _make_facing_shove_hand(hero_cards=("Ah", "Ad"), eff_bb=12.0, shover_bb=12.0)
    p = analyze_hand(en).points[0]
    assert p.best_action == "call" and p.zone == "strict"
    assert p.assumption is None

def test_no_llm_and_no_result_bias():
    # вскрытые карты соперника не влияют на вердикт: пересунем вскрытие — вердикт тот же
    en1 = _make_hu_shove_hand(hero_cards=("Ah", "Ad"), eff_bb=10.0)
    en2 = en1.model_copy(deep=True); en2.hand.showdowns = []
    assert analyze_hand(en1).points[0].ev_diff_bb == analyze_hand(en2).points[0].ev_diff_bb
```

`_make_hu_shove_hand` — конструктор синтетического `EnrichedHand` (2 игрока, блайнды 1/2 условных единиц bb-масштаба, Hero пушит all-in): собрать CanonicalHand руками + `enrich`.

- [ ] **Step 2: падают.** Step 3: реализация:
  - `classify`: pushfold_unopened — никто не вложился добровольно до Hero и eff ≤ 15bb; pushfold_facing_shove — перед Hero олин и eff ≤ 15bb (либо колл = олин Hero); прочее префлоп — preflop_other (v1: вердикт «не размечен», ev_diff 0, в отчёт не ранжируется); постфлоп — postflop (v1: пропускается, зона придёт ступенью 2 порядка разработки).
  - pushfold_unopened, `live_total == 2`: `nash_hu(eff_bb)` — равновесие, зона `strict`, `tools=["nash_hu"]`.
  - pushfold_unopened, `live_total >= 3`: `shove_ev_bb` с моделью коллеров (колл-сторона `nash_hu` по глубине каждого игрока позади) → `best_action`; затем bracket-тест: тот же расчёт с `BRACKET_TIGHT` и `BRACKET_WIDE`; зона и причина — из `zone_for(best_tight, best_wide, live_total=...)`. При `assuming` заполняется `Assumption(range=<модель среднего коллера>, source="model:nash_hu_call", note="колл-диапазоны игроков позади смоделированы: мультивей-равновесия в v1 нет")`. В `detail`: `{"method": "subset_enumeration", "bracket": "stable"|"unstable", "branches": N}`.
  - pushfold_facing_shove: `call_shove_ev_bb` против диапазона шовера (push-сторона `nash_hu` по его глубине при `live_total == 2` — зона `strict`; иначе как модель + bracket-тест по узкой/широкой моделям шова). best = call при EV > 0, иначе fold.
  - `assumption` заполняется **тогда и только тогда**, когда зона `assuming` — инвариант проверяется тестом.
  - `rank_points`: сортировка по ev_diff возрастанию (самая дорогая потеря первой); `total_ev_loss_bb` = сумма отрицательных.
- [ ] **Step 4: зелёные.** Step 5: Commit — `feat: префлоп-анализ пуш-фолд зоны строго + оценщик/ранжирование ошибок`

---

### Task 13: Префлоп-скан турнира

**Files:**
- Create: `src/harness/analysis/scan.py`
- Test: `tests/test_scan.py`

**Interfaces:**
- Produces: `scan_tournament(enriched: list[EnrichedHand]) -> ScanSummary`; `ScanSummary(BaseModel): hands_total: int; hands_with_decision: int; items: list[ScanItem]; total_loss_bb: float`; `ScanItem(BaseModel): hand_no: str; hand_index: int | None; hero_class: str; spot: SpotKind; action_taken: str; best_action: str; ev_diff_bb: float; zone: Zone` — items отсортированы по цене, только расхождения (ev_diff < −0.1bb); `zone` переносится из `PointVerdict` и показывается в сводке (мультивей-шовы в сводке не выдаются за точный расчёт). Формулировка «расхождение», не «ошибка» — честность из архитектуры.

- [ ] **Step 1: тесты**

```python
from harness.analysis.scan import scan_tournament
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file
from tests.conftest import FIXTURE_DAILY

def test_scan_daily_classic_runs():
    ens = [enrich(normalize(r)) for r in parse_file(FIXTURE_DAILY.read_text("utf-8"), "d")]
    s = scan_tournament(ens)
    assert s.hands_total == 146
    assert 0 < s.hands_with_decision <= 146          # префильтр отсёк тривиальные фолды
    assert all(s.items[i].ev_diff_bb <= s.items[i + 1].ev_diff_bb for i in range(len(s.items) - 1))
    assert all(it.ev_diff_bb < -0.1 for it in s.items)

def test_prefilter_cheap(monkeypatch):
    import harness.analysis.tools.equity as eq
    calls = {"n": 0}
    real = eq.equity_vs_range
    monkeypatch.setattr(eq, "equity_vs_range",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or real(*a, **k))
    # синтетика: HU 10bb, Hero на BTN фолдит 32o без добровольных вложений; Нэш согласен (фолд)
    en = _make_hu_fold_hand(hero_cards=("3c", "2d"), eff_bb=10.0)  # хелпер в этом же файле
    s = scan_tournament([en])
    assert s.items == []          # расхождения нет
    assert calls["n"] == 0        # EV не считался — сработал префильтр (SCALING §3)
```

- [ ] **Step 2: падают.** Step 3: реализация — цикл `analyze_hand` по рукам с префильтром до расчёта: если единственная точка Hero — фолд без добровольных вложений и лукап (Нэш/чарт) тоже говорит фолд с весом ≥ 0.9 — точка закрывается без EV-расчёта (рычаг SCALING.md §3). Скан — чистый код, ноль LLM; тяжёлые эквити прогоняются с меньшими итерациями (50k) — точности для ранжирования достаточно, в `detail` пометка.
- [ ] **Step 3b:** прогнать скан по обоим файлам локально, глазами посмотреть топ-5 расхождений на адекватность (ручная проверка владельцем — первые e2e-кандидаты в eval-датасет).
- [ ] **Step 4: зелёные.** Step 5: Commit — `feat: префлоп-скан турнира (префильтр, ранжирование расхождений)`

---

### Task 14: Память — модели БД, миграция 0001, testcontainers

**Files:**
- Create: `src/harness/memory/models.py`, `src/harness/memory/repos.py`, `alembic.ini`, `migrations/env.py`, `migrations/versions/0001_initial.py`
- Modify: `tests/conftest.py` (pg-фикстура)
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: SQLAlchemy-модели всех таблиц спеки §6 (`players, invites, sessions, tournaments, hands, analyses, notes, eval_cases, jobs, traces, llm_calls, calc_cache`); `async_session_factory(dsn)`; репозитории: `PlayersRepo.get_or_create(tg_user_id)`, `SessionsRepo.active_or_create(player_id)`, `HandsRepo.save_raw/save_canonical/save_enriched/get(hand_id)`, `AnalysesRepo.save/get_by_hand`, `EvalCasesRepo.add`; conftest-фикстура `pg` (testcontainers, scope="session") и `db` (чистая схема через `alembic upgrade head` на контейнер + транзакционный откат на тест).
- Зависимости: `uv add "sqlalchemy[asyncio]>=2.0.30" asyncpg alembic "psycopg[binary]"` и `uv add "testcontainers[postgres]" --group dev`.

- [ ] **Step 1: тесты**

```python
async def test_migration_applies(pg):            # upgrade head на настоящем PG16
    ...

async def test_hand_artifacts_roundtrip(db):
    raw = RawHand.model_validate(make_min_raw())  # хелпер задачи 2
    hid = await HandsRepo(db).save_raw(session_id=..., raw=raw)
    got = await HandsRepo(db).get(hid)
    assert got.raw == raw and got.canonical is None   # nullable-колонки = чекпоинты

async def test_jobs_required_fields(db):
    with pytest.raises(IntegrityError):
        await db.execute(insert(Job).values(type="hh_scan", status="queued", payload={}))
        await db.commit()                              # session_id NOT NULL — спека §6
```

- [ ] **Step 2: падают.** Step 3: модели — колонки по табл. §6 спеки дословно (jsonb → `JSONB`, статусы — `String` + CHECK, индексы: `jobs(status, priority, created_at)`, `jobs(player_id, status)`, `hands(session_id)`, `llm_calls(started_at)`, `eval_cases(kind)`); `0001_initial` — autogenerate + ручная правка; alembic URL — sync `postgresql+psycopg://`, рантайм — `postgresql+asyncpg://`.
- [ ] **Step 4: зелёные (реальный PG).** Step 5: Commit — `feat: схема БД v1 (12 таблиц), миграция 0001, репозитории, testcontainers`

---

### Task 15: Очередь jobs

**Files:**
- Create: `src/harness/platform/queue.py`
- Test: `tests/test_queue.py`

**Interfaces:**
- Produces: `JobsQueue(session_factory)`: `enqueue(*, type: str, player_id: int, session_id: int, payload: dict, priority: int = 100) -> int`; `claim(worker_id: str) -> Job | None`; `complete(job_id)`; `fail(job_id, error: str)`; `await_user(job_id, resume_payload: dict)`; `resume(job_id)` (awaiting_user → queued); `reap(older_than_minutes: int = 10) -> int`. Статусы: `queued|running|awaiting_user|done|failed`.

- [ ] **Step 1: тесты (реальный PG)**

```python
async def test_claim_atomic_two_workers(db_factory):
    q = JobsQueue(db_factory)
    jid = await q.enqueue(type="hh_scan", player_id=1, session_id=1, payload={})
    a, b = await asyncio.gather(q.claim("w1"), q.claim("w2"))
    assert sorted([a is not None, b is not None]) == [False, True]   # взял ровно один

async def test_per_player_serialization(db_factory):
    q = JobsQueue(db_factory)
    await q.enqueue(type="screenshot_analyze", player_id=1, session_id=1, payload={})
    await q.enqueue(type="screenshot_analyze", player_id=1, session_id=1, payload={})
    j1 = await q.claim("w1")
    assert await q.claim("w2") is None            # у игрока уже running
    await q.complete(j1.id)
    assert (await q.claim("w2")) is not None       # теперь можно

async def test_awaiting_user_not_active(db_factory):
    q = JobsQueue(db_factory)
    await q.enqueue(type="screenshot_analyze", player_id=1, session_id=1, payload={})
    j1 = await q.claim("w1")                       # первая задача игрока -> running
    await q.await_user(j1.id, resume_payload={"station": "engine"})
    await q.enqueue(type="screenshot_analyze", player_id=1, session_id=1, payload={})
    assert (await q.claim("w2")) is not None       # awaiting_user не блокирует — спека §8.1

async def test_reap_returns_stuck_running(db_factory):
    ...  # руками поставить locked_at в прошлое, reap() -> статус queued, attempts+1

async def test_fail_after_max_attempts(db_factory):
    ...  # attempts достиг max_attempts -> статус failed
```

- [ ] **Step 2: падают.** Step 3: реализация — SQL захвата одним стейтментом:

```sql
UPDATE jobs SET status='running', locked_by=:w, locked_at=now(), attempts=attempts+1
WHERE id = (
  SELECT j.id FROM jobs j
  WHERE j.status = 'queued'
    AND NOT EXISTS (SELECT 1 FROM jobs r
                    WHERE r.player_id = j.player_id AND r.status = 'running')
  ORDER BY j.priority, j.created_at
  FOR UPDATE SKIP LOCKED LIMIT 1
) RETURNING *
```

`reap`: `running` старше порога → `queued` (attempts уже инкрементирован при claim; если attempts > max_attempts → `failed`). Приоритеты: screenshot_analyze/deep_dive = 100, hh_scan = 200, eval_run = 900.

- [ ] **Step 4: зелёные.** Step 5: Commit — `feat: Postgres-очередь (SKIP LOCKED, сериализация по игроку, awaiting_user, reaper)`

---

### Task 16: llm() фасад + кросс-процессный лимитер

**Files:**
- Create: `src/harness/platform/llm.py`, `src/harness/platform/limiter.py`, `src/harness/platform/config.py`
- Test: `tests/test_llm_facade.py`

**Interfaces:**
- Produces: `class LLM: async def __call__(self, purpose: Literal["vision_extract", "verdict_text"], schema: type[T], *, prompt: str, images: Sequence[bytes] = ()) -> tuple[T, CallMeta]`; `CallMeta(model: str, tokens_in: int, tokens_out: int, cost_usd: float | None, latency_ms: int)`; `Config` из env: `LLM_VISION_MODEL`, `LLM_VERDICT_MODEL` (строки вида `"anthropic:claude-sonnet-..."` — формат PydanticAI), `LLM_MAX_CONCURRENCY` (K слотов), `LLM_MAX_PER_MINUTE`, `DATABASE_URL`, `TELEGRAM_TOKEN`.
- Зависимость: `uv add pydantic-ai` (в основные).

- [ ] **Step 1: тесты**

```python
async def test_limiter_serializes_across_connections(db_factory):
    lim = PgLimiter(db_factory, max_concurrency=1, max_per_minute=1000)
    order = []
    async def hold(tag):
        async with lim.slot():
            order.append(f"{tag}-in"); await asyncio.sleep(0.2); order.append(f"{tag}-out")
    await asyncio.gather(hold("a"), hold("b"))
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])
    # advisory-локи — кросс-СОЕДИНЕНИЕ, значит и кросс-процессно

async def test_rate_window_blocks(db_factory, monkeypatch):
    ...  # вставить LLM_MAX_PER_MINUTE строк llm_calls со started_at=сейчас -> следующий вызов ждёт

async def test_facade_validates_and_logs(db_factory):
    from pydantic_ai.models.test import TestModel
    llm = LLM(cfg, db_factory, model_override=TestModel())   # тестовая модель PydanticAI
    class Out(BaseModel): text: str
    out, meta = await llm("verdict_text", Out, prompt="скажи привет")
    assert isinstance(out, Out)
    rows = await fetch_all(db_factory, "select purpose, status from llm_calls")
    assert rows == [("verdict_text", "ok")]

async def test_retry_on_schema_error_then_fail(db_factory):
    ...  # FunctionModel, отдающая мусор: 1 ретрай, затем исключение; в llm_calls двe строки со status='schema_error'
```

- [ ] **Step 2: падают.** Step 3: реализация:
  - `PgLimiter.slot()` — async contextmanager: выделенное соединение, перебор `SELECT pg_try_advisory_lock(0x4C4C4D, i)` для i in range(K), при неудаче `sleep(0.05..0.15 джиттер)` и повтор; в `finally` — `pg_advisory_unlock` на том же соединении. Темп: перед захватом `SELECT count(*) FROM llm_calls WHERE started_at > now() - interval '60 seconds'`; если ≥ лимита — sleep с джиттером до входа в окно. Строка `llm_calls` вставляется **до** вызова модели (status='started'), обновляется по завершении (спека §7: in-flight входят в окно).
  - `LLM.__call__`: purpose → модель из конфига; `pydantic_ai.Agent(model, output_type=schema)`; images → `BinaryContent`; бэкофф на 429/5xx: 3 попытки, `2^n + jitter`; локальный `asyncio.Semaphore` как предфильтр (не источник истины).
- [ ] **Step 4: зелёные.** Step 5: Commit — `feat: фасад llm() с кросс-процессным лимитером (advisory-локи + окно по llm_calls)`

---

### Task 17: presentation — единый голос продукта

**Files:**
- Create: `src/harness/presentation/messages.py`, `src/harness/presentation/keyboards.py`
- Test: `tests/test_presentation.py`

**Interfaces:**
- Produces (чистые функции, ни ТГ-API, ни БД — по «правилу единого голоса» §4):
  - `Msg(BaseModel): text: str; buttons: list[list[Btn]] = []` · `Btn(BaseModel): text: str; callback_data: str`
  - `progress_text(station: Literal["parse","validate","analyze","explain"]) -> str` — «Читаю стол… / Проверяю руку… / Считаю эквити… / Формулирую…»
  - `scan_summary_msg(s: ScanSummary, quota_left: int, quota_total: int) -> Msg` — топ расхождений с ценой в bb, под каждым `Btn("разобрать", f"deep:{hand_no}")`; формулировки «расхождение», не «ошибка»; строки с `zone == "assuming"` несут пометку «по модели диапазонов» (тест: пометка есть у assuming-строк и отсутствует у strict)
  - `deep_dive_msg(res: AnalysisResult, elapsed_s: int, zone: Zone, quota_left: int, quota_total: int) -> Msg` — точки решения числами (до задачи 21 — без LLM-текста) + статус-строка `⏱ {N}с · зона: {строго|предполагая} · разборов {осталось}/{всего} за 24 ч` + кнопки `Диапазоны | Подробнее | Не согласен`
  - `escalation_msg(field: str, question: str, options: list[str]) -> Msg` — варианты + `Btn("ввести вручную", ...)`
  - `failed_msg(reason_public: str) -> Msg`, `quota_exceeded_msg(hours_to_free: int) -> Msg`
- Тон: без жаргона и внутренней кухни (SESSIONS_UX); строка стоимости — только под dev-флагом (аргумент `dev_line: str | None = None`).

- [ ] **Step 1: тесты** — на каждый конструктор: текст содержит ожидаемые числа/слова («−2.3 bb», «за 24 ч», «строго»), кнопки с правильными callback_data, «ошибка» не встречается в текстах скана, dev-строка появляется только при dev_line.
- [ ] **Step 2–4: падают → реализация → зелёные.** Step 5: Commit — `feat: presentation — все сообщения игроку в одном модуле`

---

### Task 18: worker — оркестрация, чекпоинты, трейс

**Files:**
- Create: `src/harness/worker/pipeline.py`, `src/harness/worker/main.py`, `src/harness/platform/trace.py`
- Test: `tests/test_worker_pipeline.py`

**Interfaces:**
- Consumes: очередь (15), память (14), presentation (17), конвейер (4–13), `llm` (16 — пока не используется станциями v1-HH).
- Produces: `run_job(job: Job, deps: Deps) -> None`; `Deps(db_factory, queue, sender, llm, clock)`; `Sender` — протокол `send(chat_id, msg: Msg) -> int (message_id)`, `edit(chat_id, message_id, msg: Msg)`; `Trace` — контекстменеджер `span(name)`, собирает spans, `flush(job_id)` в `traces`.
- Станции по типам: `hh_scan`: сохранить руки (raw → canonical → enriched чекпоинтами в `hands`) → `scan_tournament` → `tournaments.scan_summary` → отправить `scan_summary_msg`. `deep_dive`: найти руку по hand_no → `analyze_hand` → `analyses` → `deep_dive_msg`. Прогресс: `edit` одного сообщения по станциям; message_id прогресса и результата — в `jobs.payload` (идемпотентность — спека §8.2).

- [ ] **Step 1: тесты (реальный PG, FakeSender со списком отправленного)**

```python
async def test_hh_scan_end_to_end(db_factory, fake_sender):
    jid = await enqueue_hh_scan(FIXTURE_DAILY, player_id=1, session_id=1)
    job = await queue.claim("w1"); await run_job(job, deps)
    assert (await job_status(jid)) == "done"
    assert fake_sender.edits          # прогресс редактировался
    final = fake_sender.sent[-1]
    assert "bb" in final.text and any(b.callback_data.startswith("deep:") for row in final.buttons for b in row)
    assert await count(db_factory, "hands") == 146          # артефакты записаны
    assert await count(db_factory, "traces") == 1

async def test_resume_skips_done_stations(db_factory, fake_sender, monkeypatch):
    # предзаполнить hands.raw/canonical/enriched для всех рук, уронить parse_file при вызове
    monkeypatch.setattr("harness.parsers.hh_parser.parse_file", _boom)
    job = await queue.claim("w1"); await run_job(job, deps)   # должен пройти без парсера
    assert (await job_status(job.id)) == "done"               # чекпоинты работают — спека §8.2

async def test_send_idempotent(db_factory, fake_sender):
    # payload уже содержит result_message_id -> повторный run_job не шлёт дубль
    ...

async def test_failure_marks_failed_and_notifies(db_factory, fake_sender, monkeypatch):
    ...  # analyze_hand кидает -> статус failed (после max_attempts), игроку failed_msg
```

- [ ] **Step 2: падают.** Step 3: реализация — `main.py`: цикл `claim → run_job` в N корутин (конфиг `WORKER_CONCURRENCY`), фоновая корутина `reap()` раз в минуту; скан-расчёты — через `asyncio.get_running_loop().run_in_executor(process_pool, ...)` (CPU не блокирует loop — спека §2); structlog-контекст `job_id/hand_no`.
- [ ] **Step 4: зелёные.** Step 5: Commit — `feat: воркер — оркестрация станций, чекпоинты, трейс, идемпотентная отправка`

---

### Task 19: bot — приём, молчаливая сессия, HH-путь

**Files:**
- Create: `src/harness/bot/main.py`, `src/harness/bot/router.py`, `src/harness/bot/handlers.py`
- Test: `tests/test_bot_handlers.py`

**Interfaces:**
- Consumes: память (14), очередь (15), presentation (17).
- Produces: обработчики-функции с внедрёнными зависимостями (aiogram — тонкая обвязка поверх них, сама не тестируется):
  - `handle_document(deps, tg_user_id: int, file_bytes: bytes, filename: str) -> Msg` — сохранить файл на диск (`DATA_DIR/hh/{hash}.txt`), `PlayersRepo.get_or_create`, `SessionsRepo.active_or_create` (**молчаливое создание сессии — здесь**, спека §6/§13 шаг 6), создать `tournaments`-строку, `enqueue(type="hh_scan", ...)`, вернуть подтверждение;
  - `handle_deep_dive_callback(deps, tg_user_id, hand_no: str) -> Msg | None` — `enqueue(type="deep_dive")`;
  - `handle_new_session(deps, tg_user_id) -> Msg` — `/new`: закрыть активную, открыть новую;
  - `check_quota(deps, player_id) -> QuotaCheck(allowed: bool, left: int, total: int, hours_to_free: int)` — SQL-счётчик интерактивных задач за 24 ч (спека §9), при `not allowed` — `quota_exceeded_msg`;
  - `bot/main.py` — aiogram 3: Router, `F.document` → скачивание → `handle_document`, `CallbackQuery(F.data.startswith("deep:"))`, `/new`, `/start`; long polling.
- Зависимость: `uv add aiogram`.
- Фото (`F.photo`) в этой задаче отвечает вежливой заглушкой «скоро» через presentation — vision приходит задачей 22. Инвайты — задача 23; до неё бот отвечает только `players`-записям, создаваемым без ограничений (закрытый догфудинг).

- [ ] **Step 1: тесты** — на функции-обработчики с реальным PG и фейковой очередью-обёрткой:

```python
async def test_document_creates_session_silently(db_factory):
    msg = await handle_document(deps, tg_user_id=777, file_bytes=FIXTURE_DAILY.read_bytes(),
                                filename="t.txt")
    s = await fetch_one(db_factory, "select * from sessions")     # сессии не было — создана молча
    j = await fetch_one(db_factory, "select * from jobs")
    assert j["type"] == "hh_scan" and j["session_id"] == s["id"]   # NOT NULL выполняется по построению

async def test_second_file_same_session(db_factory): ...           # активная сессия переиспользуется
async def test_new_closes_and_opens(db_factory): ...
async def test_quota_window_24h(db_factory):
    # 3 задачи: 2 свежих, 1 старше 24ч; quota_daily=2 -> запрещено; старение освобождает
    ...
```

- [ ] **Step 2–4: падают → реализация → зелёные.** Step 5: Commit — `feat: бот — HH-путь с молчаливой сессией, [разобрать], квота 24ч`

---

### Task 20: Деплой — Docker Compose, отдельный migrate

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.env.example`, `README.md` (раздел «Деплой»)

- [ ] **Step 1: Dockerfile** — `FROM python:3.12-slim`, uv из образа astral, `uv sync --frozen --no-dev`, один образ, entrypoint выбирается командой сервиса.
- [ ] **Step 2: docker-compose.yml**

```yaml
services:
  postgres:
    image: postgres:16
    volumes: ["pgdata:/var/lib/postgresql/data"]
    environment: { POSTGRES_DB: harness, POSTGRES_USER: harness, POSTGRES_PASSWORD: "${PG_PASSWORD}" }
  migrate:
    build: .
    profiles: ["tools"]                       # не стартует с up
    command: ["uv", "run", "alembic", "upgrade", "head"]
    depends_on: [postgres]
  bot:
    build: .
    command: ["uv", "run", "python", "-m", "harness.bot.main"]
    depends_on: [postgres]
    env_file: [.env]
  worker:
    build: .
    command: ["uv", "run", "python", "-m", "harness.worker.main"]
    depends_on: [postgres]
    env_file: [.env]
volumes: { pgdata: {} }
```

Порядок деплоя (README, дословно из спеки §12): `docker compose build && docker compose run --rm migrate && docker compose up -d`; масштаб: `docker compose up -d --scale worker=2`. Entrypoint'ы `bot`/`worker` миграций не катят — гонка исключена по построению. Бэкап: cron `pg_dump` строкой в README.

- [ ] **Step 3:** локальная проверка полного цикла: `compose build → run --rm migrate → up -d → закинуть HH-файл боту → получить сводку скана` (ручная, результат зафиксировать в PR-описании).
- [ ] **Step 4: Commit** — `feat: деплой — compose с отдельным шагом миграций`

> **Веха:** после задачи 20 продукт полезен (скан турнира в ТГ) при нуле LLM-вызовов — спека §13, шаг 6.

---

### Task 21 (крупнее, спека §13 шаг 7): Изложение — текст вердикта, рендер матрицы, verdict-evals

**Files:**
- Create: `src/harness/explanation/verdict_text.py`, `src/harness/explanation/range_render.py`, `src/harness/platform/eval_runner.py`, `evals/verdict/checks.py`, `evals/verdict/cases/` — эталоны
- Modify: `src/harness/worker/pipeline.py` (станция explain для `deep_dive`), `src/harness/presentation/messages.py` (вердикт с LLM-текстом), `tests/` — новые файлы по образцу задач 9–18
- Зависимость: `uv add cairosvg`

**Interfaces:**
- `render_range_png(rng: Range, title: str) -> bytes` — сетка 13×13: ranks `AKQJT98765432`; клетка (i, j): i==j → пара; i<j → `ranks[i]+ranks[j]+"s"`; i>j → `ranks[j]+ranks[i]+"o"`; вес 0..1 — вертикальная доля заливки клетки (SVG rect поверх фона), SVG → PNG cairosvg. Тест: в SVG ровно 169 подписанных клеток, вес 0.5 даёт rect половинной высоты, картинка > 10 КБ.
- `class VerdictTextOut(BaseModel): points: list[PointText]; summary: str` · `PointText(dp_index: int, verdict_label: Literal["ok", "mistake", "marginal"], text: str)` — **метка вердикта отдаётся структурно**, текст только излагает; промпт получает выжимку `AnalysisResult` (числа и вердикты), не весь `EnrichedHand` (SCALING §1: сжатие).
- `verdict_text(llm: LLM, res: AnalysisResult) -> VerdictTextOut` — единственный второй вызов LLM в системе.
- Verdict-evals (EVALS этаж 3) — `evals/verdict/checks.py`, три проверки кодом:
  1. числа: `re.findall(r"-?\d+(?:[.,]\d+)?", text)` ⊆ множества чисел из `AnalysisResult` (с округлением до 1 знака);
  2. знак: `verdict_label` каждого PointText равен метке ядра (`ok` при ev_diff ≥ −0.1, `mistake` при < −0.5, иначе `marginal`) — расхождение = провал;
  3. допущение: если у точки `zone == "assuming"` — в тексте есть слова допущения («если », «предполагая», «допущени») — паттерн-проверка.
- `eval_runner` CLI: `uv run python -m harness.platform.eval_runner verdict` — гонит кейсы, печатает диффы; обязателен при смене `LLM_VERDICT_MODEL`/промпта.

Шаги внутри — по стандартному циклу задач 9–18 (тест → красный → реализация → зелёный → коммит на каждый из трёх модулей). Worker: станция explain пишет `analyses.verdict_text` + PNG на диск (чекпоинт), presentation собирает финальное сообщение с текстом LLM.

---

### Task 22 (крупнее, спека §13 шаг 8): Vision-парсер и эскалации

> **Плановая точка детализации:** перед стартом задача разбивается на подзадачи по записке спайка (задача 8). Ниже — фиксированный каркас.

**Files:**
- Create: `src/harness/parsers/vision_adapter.py`, `evals/vision/` (датасет пар скрин → эталонный RawHand, старт ~20–30, разметка владельцем), `harness/platform/eval_runner.py` — подкоманда `vision`
- Modify: `bot/handlers.py` (`F.photo` → сохранить, `image_hash`, enqueue `screenshot_analyze`), `worker/pipeline.py` (станция vision + путь эскалации), `bot` callback-обработчики ответов эскалации

**Interfaces:**
- `vision_extract(llm: LLM, image: bytes) -> RawHand` — схема выхода = RawHand-совместимая (provenance="screenshot", VisionMeta с confidence/needs_review/nicknames/bounties);
- эскалация по спеке §8.3 дословно: worker шлёт `escalation_msg`, `queue.await_user(job_id, resume_payload)`; bot ловит callback → `EvalCasesRepo.add(kind="vision_field", ...)` (ground truth!) → патч `hands.raw` → сброс `canonical/enriched` → `queue.resume(job_id)`; «ввести вручную» — FSM-ввод числа;
- vision-eval (EVALS этаж 2): метрики точности по типам полей, отдельно критичные (карты Hero, суммы), % прошедших валидатор без эскалации; прогон при каждой смене vision-модели/промпта. Датасет собирается ДО выката vision в прод (EVALS §порядок, п. 3).
- Кнопка «Не согласен» (уже в presentation) начинает писать `eval_cases(kind="verdict_dispute")` — вход e2e-датасета (этаж 4); `eval_runner` получает подкоманду `e2e`: прогон конвейера по подтверждённым разборам из `eval_cases`, диффы вердиктов (EVALS: «настраивается, когда конвейер ходит насквозь» — то есть здесь).

---

### Task 23 (крупнее, спека §13 шаг 9): Сессии-UI, «Мои лики», заметки, инвайты, квоты-UI

> **Плановая точка детализации:** после первого догфудинга HH-пути (задачи 19–20) — открытые UX-хвосты SESSIONS_UX (агрегат прошлой сессии, формат «Моих ликов», `/end`) закрываются решениями владельца и разбиваются на подзадачи.

**Files:**
- Modify: `bot/handlers.py`, `presentation/*`, `memory/repos.py`
- Create: `bot/menus.py`

**Каркас:**
- нижнее меню (постоянная клавиатура SESSIONS_UX): Сессии · Турнир (HH) · Мои лики · Заметки · Настройки · Help;
- «Сессии»: список со сводками (агрегат: турниров/рук, суммарная цена расхождений bb, повторяющийся лик — SQL по `analyses.result.points` группировкой по `spot`), «Начать новую» = `/new`;
- «Мои лики»: сквозная группировка PointVerdict по spot/классам рук за всю историю, частота × цена;
- «Заметки»: список оппонентов с цветом/текстом, CRUD (данные только от vision — identity=nick);
- инвайты: `/start <code>` deep-link, `invites.used_by` → создание `players`; без кода — вежливый отказ (presentation);
- «Настройки»: dev-флаг себе выключить нельзя, видимого минимума достаточно (v1: только «о боте»); квотные сообщения уже в presentation.

---

## Порядок, параллельность, обновление плана

- Строгая последовательность: 1 → 2 → 3 → 4 → 5 → 6 → 7; 14 → 15 → 16/17 → 18 → 19 → 20; 21 → 22 → 23.
- Параллелить можно: задачу 8 (спайк) — с любой из 4–13; задачи 9–11 — между собой; 16 и 17 — между собой.
- Чекпоинты ревью владельцем: после 7 (сетка зелёная), после 13 (скан глазами на своих турнирах), после 20 (веха: продукт в ТГ), перед стартом 21/22/23 (точки детализации).
- Правки плана — по правилу из брейншторма: изменения интерфейсов или порядка проходят через владельца и коммитятся; пометка выполненных задач — в этом файле (чекбоксы).
