"""Скриншот -> `RawHand`: единственное место, где зрение встречается с конвейером.

Vision — второе из двух мест, где в системе живёт LLM (CLAUDE.md), и граница
проекта проходит прямо внутри этого файла. Модель заполняет `VisionReading` —
наблюдения в единицах экрана; всё остальное делает код:

* **делит пул анте** на число игроков (реестр B2 — модель этого не делает);
* **переводит единицы** к одной (реестр A4 — приведённое моделью число выглядит
  безупречно, и проверить его нечем);
* **восстанавливает рассадку** из порядка хода в логе префлопа и сверяет её с
  фишкой дилера (реестр D3);
* **восстанавливает стартовые стеки**: экран показывает остаток ПОСЛЕ ставок и
  ДО раздачи банка (измерено на двух руках фикстуры до фишки), поэтому
  `стартовый = показанный + анте + вложенное − возвращённое`;
* **опознаёт героя** по нику из профиля, а не по подсветке на экране;
* **считает контрольные суммы** (`vision_checks`) и решает, звать ли дорогую
  модель.

**Каскад — три ступени, и каждая записана.** Чтение дешёвой моделью
(`LLM_VISION_MODEL`); провал любой контрольной суммы — повтор на дорогой
(`LLM_VISION_FALLBACK_MODEL`); провал и там — вопрос игроку. Ступени лежат в
`VisionMeta.hops`, воркер переносит их в трейс. Цель сформулирована владельцем
как «высокая доля с первого раза при нуле тихих ошибок»: вопрос игроку — не
провал, а предохранитель, потому что скрин по архитектуре гипотеза, а не факт.

**Ников в промпте нет.** Ник героя, ожидаемый банк, известный уровень блайндов —
всё это подсказывает ОТВЕТ, а подсказанный ответ невозможно отличить от
прочитанного. В промпт идёт только то, что снимает неоднозначность: определения
полей и единицы (реестр, «Почему ник НЕ передаётся»).

**Метки игроков анонимны.** В `RawHand` едут `S1…Sn` и `Hero`; настоящие ники
лежат отдельно, в `VisionMeta.nicknames`, откуда их берут заметки на игроков
(задача 23). Так весь конвейер ниже работает с теми же обезличенными метками,
что и HH-путь, и ник не может утечь ни в отчёт, ни в тест, ни в лог.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from harness.contracts import (
    ActionKind,
    Collected,
    Completeness,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    SeenAction,
    SeenPlayer,
    ShowdownEntry,
    Street,
    Uncalled,
    Unit,
    VisionCheck,
    VisionHop,
    VisionMeta,
    VisionReading,
)
from harness.parsers.vision_checks import (
    CHECK_BUTTON,
    POT_TOLERANCE_BB,
    button_check,
    cards_check,
    equity_check,
    match_hero,
    positions_check,
    pot_check,
    seats_check,
    within_tolerance,
)

__all__ = [
    "CHECK_TRUNCATION",
    "HERO_LABEL",
    "PromptUnavailable",
    "VisionLLM",
    "VisionOutcome",
    "apply_vision_answer",
    "read_prompt",
    "reading_to_raw",
    "vision_extract",
]

_T = TypeVar("_T", bound=BaseModel)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "vision.md"

HERO_LABEL = "Hero"

# Условная «фишка» для экранов, где всё напечатано в больших блайндах, а самих
# фишек нет нигде (экспорт истории именно таков). Контракт `RawHand` требует
# целых сумм, а единственная величина, к которой на таком экране можно всё
# привести, — сам большой блайнд. Десять тысяч единиц на блайнд означают, что
# напечатанные два знака после запятой переживают перевод без потерь, а деление
# пула анте на восьмерых не упирается в округление.
_BB_UNIT = 10_000

# Экрана времени раздачи не печатает нигде, а `RawHand.timestamp` обязателен.
# Заведомо невозможная дата честнее правдоподобной: подставить «сейчас» значило
# бы записать в поле факта догадку, и отличить её потом было бы нечем.
_NO_TIMESTAMP = datetime(1, 1, 1, tzinfo=UTC)

# Экран печатает суммы с двумя знаками и ОБРЕЗАЕТ их, а не округляет (сверено с
# текстом рума: 87 589 фишек при блайнде 5 000 напечатаны как 17.51, хотя это
# 17.5178). Поэтому восстановленный стартовый стек бывает на сотую ББ меньше
# того, что игрок заведомо вложил, и рука становится непроигрываемой: движок
# упирается в «поставил больше, чем имел». Такой стек поднимается до вложенного —
# это поправка на известную обрезку экрана, а не догадка о деньгах, и она
# ограничена сверху: больше `_TRUNCATION_LIMIT` означает, что дело не в обрезке,
# и расхождение обязано дойти до игрока.
_TRUNCATION_LIMIT_BB = 0.05
CHECK_TRUNCATION = "stacks"

# Метка ступени каскада, вернувшей пустую схему (см. `VisionReadFailed`).
_EMPTY_READING = "пустое чтение"

# Ступень каскада, остановленная незавершённой рукой. Пишется в `VisionHop.error`
# рядом с `_EMPTY_READING` и `not_a_hand`: все три означают «дальше по каскаду не
# пошли», и трейс обязан показывать, по какой из трёх причин.
_HAND_IN_PROGRESS = "рука ещё идёт"

_SB_LABELS = {"sb", "мб", "мблайнд", "small blind"}
_BB_LABELS = {"bb", "бб", "ббл", "big blind"}

_BOARD_SPLIT: dict[int, dict[Street, int]] = {
    3: {Street.FLOP: 3},
    4: {Street.FLOP: 3, Street.TURN: 1},
    5: {Street.FLOP: 3, Street.TURN: 1, Street.RIVER: 1},
}


class PromptUnavailable(RuntimeError):
    """Файла промпта нет на диске — работать без него нельзя, и молчать тоже.

    Vision-промпты закрыты политикой публикации (CLAUDE.md, `.githooks/pre-push`):
    в публичном клоне репозитория этого файла НЕТ. Пустой промпт вместо файла дал
    бы модели пустое задание и произвольную структуру в ответ, а контрольные
    суммы на пустом чтении молчат — то есть тихую деградацию вместо отказа.

    Тот же тип и та же причина, что у `harness.explanation.verdict_text`
    (задача 21): отдельный класс, потому что это конфигурация развёртывания, а
    не сбой модели.
    """


class VisionReadFailed(RuntimeError):
    """Экран прочитать не удалось: модель вернула пустую схему на всех ступенях.

    Не то же, что `not_a_hand`, и разница существенна. Отказ — это ОТВЕТ: модель
    посмотрела и говорит, что раздачи здесь нет, с причиной. Пустая схема — сбой:
    все поля `VisionReading` необязательны (иначе модель дописывала бы то, чего
    не видит), поэтому пустой объект проходит валидацию и от честного отказа
    неотличим ничем, кроме отсутствия причины.

    Пустое чтение поэтому повторяется — сначала на той же модели, потом на
    дорогой; и только когда пусто везде, поднимается это исключение (случай, из
    которого это правило выросло, — в отчёте прогона датасета). Показать игроку
    «это не раздача» на
    таком чтении было бы утверждением, которого никто не делал.
    """


class VisionLLM(Protocol):
    """Ровно та часть фасада `platform.llm.LLM`, которой пользуется зрение.

    Протокол, а не импорт класса: конвейерные пакеты не знают про `platform`
    (правило зависимостей CLAUDE.md), а тестам нужен двойник без сети и без
    Postgres. Тот же приём, что у `explanation.verdict_text.VerdictLLM`.
    """

    async def __call__(
        self,
        purpose: Literal["vision_extract", "vision_extract_fallback"],
        schema: type[_T],
        *,
        prompt: str,
        images: Sequence[bytes] = (),
        trace_id: int,
    ) -> tuple[_T, Any]: ...


@dataclass(frozen=True, slots=True)
class VisionOutcome:
    """Что вернуло зрение: рука, ступени каскада и список непройденных проверок.

    `raw is None` — разбирать нечего: модель отказалась (`not_a_hand`) либо рука
    на экране ещё не доиграна (`hand_in_progress`). Причины разные, и вызывающий
    их различает: у первой есть `refusal` словами модели, у второй показывать
    нечего, потому что читать экран до конца мы и не стали.

    `escalate` — рука построена, но одной из проверок она не удовлетворила и на
    дорогой модели тоже: дальше вопрос игроку, а не молчаливое «разобрали».
    """

    raw: RawHand | None
    checks: list[VisionCheck] = field(default_factory=list)
    hops: list[VisionHop] = field(default_factory=list)
    refusal: str | None = None
    # Экран прочитан, но конца раздачи на нём не видно (`Completeness.STATE`).
    hand_in_progress: bool = False

    @property
    def escalate(self) -> bool:
        return self.raw is not None and any(not check.passed for check in self.checks)

    @property
    def failed(self) -> list[VisionCheck]:
        return [check for check in self.checks if not check.passed]


def _is_empty(reading: VisionReading) -> bool:
    """Модель не прочитала ничего: ни игроков, ни отказа с причиной.

    Все поля схемы необязательны намеренно (пустое поле честнее выдуманного), и
    цена этого решения ровно здесь: пустой объект валиден. Отличить его от
    ответа можно только по содержимому — см. `VisionReadFailed`.
    """
    return not reading.players and not reading.not_a_hand


def read_prompt(path: Path = _PROMPT_PATH) -> str:
    """Текст промпта с диска — ЛЕНИВО, на вызове, а не на импорте.

    Импорт `harness.parsers` обязан работать в клоне без промптов: падать должен
    тот, кто собрался звать модель, а не тот, кто импортировал пакет ради
    HH-парсера.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptUnavailable(
            f"нет файла промпта {path} — он закрыт политикой публикации "
            f"(docs/publishing-policy.md); без него вызов модели невозможен"
        ) from exc


# --- перевод единиц ----------------------------------------------------------


def _bb_in_chips(reading: VisionReading) -> int:
    """Сколько «фишек» в одном большом блайнде для этого экрана.

    Если экран печатает блайнды в фишках (живой стол показывает уровень в шапке),
    берём напечатанное. Если всё на экране в ББ (экспорт истории), берём условную
    единицу `_BB_UNIT` — см. её комментарий.
    """
    if reading.blind_unit is Unit.CHIPS and reading.bb:
        return round(reading.bb)
    for post in reading.blinds_block:
        if post.label.strip().casefold() in _BB_LABELS and post.unit is Unit.CHIPS and post.amount:
            return round(post.amount)
    return _BB_UNIT


def _chips(value: float | None, unit: Unit | None, bb_chips: int) -> int:
    """Число экрана в общих единицах. Единица неизвестна — считаем как есть.

    Умолчание «как есть» выбрано в пользу фишек, потому что второе значение
    (`bb`) умножает на блайнд: ошибиться в сторону умножения значит завысить
    сумму в тысячи раз, ошибиться в другую — занизить её так, что несходимость
    поймает первая же контрольная сумма.
    """
    if value is None:
        return 0
    return round(value * bb_chips) if unit is Unit.BB else round(value)


# --- рассадка ----------------------------------------------------------------


def _blind_owner(reading: VisionReading, labels: set[str]) -> tuple[str | None, float | None]:
    """Ник и сумма строки колонки блайндов. Ник `None` — это строка героя."""
    for post in reading.blinds_block:
        if post.label.strip().casefold() in labels:
            return post.nickname, post.amount
    return None, None


def _first_actors(actions: Sequence[SeenAction]) -> list[str | None]:
    """Кто ходил на префлопе, в порядке лога, по одному разу на игрока.

    Порядок строк лога префлопа И ЕСТЬ порядок хода: GG печатает их сверху вниз,
    начиная с UTG и кончая большим блайндом. Игрок, действовавший дважды (рейз,
    потом колл), учитывается по первому разу — рассадка от его второго хода не
    меняется. Строка без ника — героя, и она держит своё место в ряду.
    """
    seen: list[str | None] = []
    for action in actions:
        if action.street is not Street.PREFLOP:
            continue
        key = action.nickname
        if key is not None and key in seen:
            continue
        if key is None and None in seen:
            continue
        seen.append(key)
    return seen


def _ring_from_log(reading: VisionReading, hero: str | None) -> list[str | None] | None:
    """Круг мест от малого блайнда, восстановленный по порядку хода на префлопе.

    Возвращает ники в порядке `SB, BB, UTG, …, BTN` — ровно тот порядок, по
    которому нормалайзер раздаёт позиции. `None` в списке — герой (его строку
    экспорт печатает без подписи, и в колонке блайндов его строка тоже без ника).

    `None` вместо списка — круг не сложился: в логе не все, кто сидит за столом,
    или блайнды не опознаны. Тогда рассадку восстанавливать не из чего, и это
    расхождение обязано дойти до игрока, а не быть заполненным догадкой.
    """
    sb_nick, _ = _blind_owner(reading, _SB_LABELS)
    bb_nick, _ = _blind_owner(reading, _BB_LABELS)
    if not reading.blinds_block or not reading.actions:
        return None
    blinds = [sb_nick, bb_nick]
    rest = [nick for nick in _first_actors(reading.actions) if nick not in blinds]
    ring = blinds + rest
    at_table = {p.nickname for p in reading.players}
    if {nick if nick is not None else hero for nick in ring} != at_table:
        return None
    return ring


def _ring_from_seats(reading: VisionReading) -> list[str | None] | None:
    """Круг мест на живом столе: по нумерации мест, начиная со следующего за кнопкой.

    Логи улиц там не печатаются, и порядок хода взять неоткуда — остаётся
    зрительный порядок мест по кругу плюс фишка дилера. Малый блайнд это
    следующее место после кнопки (в хедз-апе — сама кнопка, и раскладку для него
    делает нормалайзер).
    """
    seated = [p for p in reading.players if p.seat is not None]
    if len(seated) != len(reading.players) or not seated:
        return None
    ordered = sorted(seated, key=lambda p: p.seat or 0)
    button = next((i for i, p in enumerate(ordered) if p.has_button), None)
    if button is None:
        return None
    start = button if len(ordered) == 2 else (button + 1) % len(ordered)
    return [ordered[(start + i) % len(ordered)].nickname for i in range(len(ordered))]


# --- сборка руки -------------------------------------------------------------


def _uncalled_by_player(commits: dict[str, int]) -> tuple[str | None, int]:
    """Непоколленный остаток круга: превышение самой крупной ставки над второй.

    Правило холдема, а не наблюдение: то, что не покрыл никто, возвращается
    поставившему. Экран этой строки не печатает, но без неё стартовый стек не
    восстановить — на экране виден остаток УЖЕ с возвращённым (сверено с текстом
    рума: игрок с олл-ин-рейзом показан ровно на сумму возврата).
    """
    if len(commits) < 2:
        return (next(iter(commits), None), next(iter(commits.values()), 0))
    ranked = sorted(commits.items(), key=lambda item: item[1], reverse=True)
    top, second = ranked[0], ranked[1]
    return (top[0], top[1] - second[1]) if top[1] > second[1] else (None, 0)


def _street_commits(
    actions: Sequence[RawAction], posts: Sequence[Post]
) -> dict[Street, dict[str, int]]:
    """Итог, поставленный каждым игроком на каждой улице, — по правилам источника.

    `raises X to Y` задаёт итог, `calls N`/`bets N` прибавляет доплату. Те же две
    формы, что в hand history GG, и та же арифметика, что в нормалайзере
    (`normalize._canonical_actions`), включая её начальное условие: блайнд УЖЕ
    лежит на префлопе, поэтому круг начинается не с нуля. Без этого доплата
    блайнда считалась бы от нуля, а сброшенный блайнд не считался бы вовсе.

    Держать эту арифметику здесь отдельно приходится потому, что стартовые стеки
    нужны РАНЬШЕ, чем рука доходит до нормалайзера.
    """
    preflop: dict[str, int] = {}
    for post in posts:
        if post.kind in (PostKind.SMALL_BLIND, PostKind.BIG_BLIND):
            preflop[post.label] = preflop.get(post.label, 0) + post.amount
    commits: dict[Street, dict[str, int]] = {Street.PREFLOP: preflop}
    for action in actions:
        street = commits.setdefault(action.street, {})
        if action.kind is ActionKind.RAISE and action.to_amount is not None:
            street[action.label] = action.to_amount
        elif action.kind in (ActionKind.CALL, ActionKind.BET) and action.amount is not None:
            street[action.label] = street.get(action.label, 0) + action.amount
    return commits


def _cards(player: SeenPlayer) -> list[str]:
    """Карты игрока для руки: из лога, если он их даёт, иначе из-под баннера.

    Порядок предпочтения — вывод замера (реестр, «Карты отрисованы дважды»):
    чистый рендер колонки лога читается безошибочно, карта у места бывает
    перекрыта баннером WIN. Расхождение между этими двумя чтениями к этому
    моменту уже названо отдельной контрольной суммой, и молчаливый выбор
    лучшего источника её не подменяет.
    """
    return player.cards_in_log or player.cards_at_seat


def reading_to_raw(
    reading: VisionReading,
    *,
    hero_nickname: str | None,
    source_ref: str,
    image_hash: str | None = None,
) -> tuple[RawHand, list[VisionCheck]]:
    """Наблюдения -> сырая рука. Здесь и только здесь код делит, переводит и считает.

    Возвращает вместе с рукой те контрольные суммы, которые считаются по
    построенной рассадке (кнопка), — остальные считает `_run_checks` по самому
    чтению.

    `hero_nickname` — ник, ОПОЗНАННЫЙ кодом (`vision_checks.match_hero`), а не
    прочитанный моделью. `None` означает, что героя опознать не удалось: рука всё
    равно строится (её предъявят игроку вопросом), а место героя остаётся за
    строкой без подписи в логе, если она есть.
    """
    bb_chips = _bb_in_chips(reading)
    ring = _ring_from_log(reading, hero_nickname) or _ring_from_seats(reading)
    checks: list[VisionCheck] = []

    by_nick = {p.nickname: p for p in reading.players}
    if ring is not None and hero_nickname is not None:
        # Строку героя экспорт печатает без подписи — и в логе, и в колонке
        # блайндов. За столом ник у него есть, поэтому безымянное место в круге
        # замещается опознанным ником: иначе стек героя не с чем сопоставить.
        ring = [hero_nickname if nick is None else nick for nick in ring]
    if ring is None:
        ring = [p.nickname for p in reading.players]
        checks.append(
            VisionCheck(
                name=CHECK_BUTTON,
                passed=False,
                detail="рассадку не восстановить: лог префлопа не покрывает стол",
            )
        )
    else:
        marked = next((p.nickname for p in reading.players if p.has_button), None)
        derived = ring[-1] if len(ring) > 2 else ring[0]
        checks.append(
            button_check(marked, hero_nickname if derived is None else derived)
        )

    labels: dict[str | None, str] = {}
    for index, nickname in enumerate(ring, start=1):
        is_hero = nickname is None or (hero_nickname is not None and nickname == hero_nickname)
        labels[nickname] = HERO_LABEL if is_hero else f"S{index}"
    ordered_labels = [labels[nickname] for nickname in ring]

    players_count = len(ring)
    ante_pool = _chips(reading.ante_pool_shown, reading.ante_unit, bb_chips)
    ante_each = (
        _chips(reading.ante_per_player_shown, reading.ante_unit, bb_chips)
        if reading.ante_per_player_shown is not None
        else (round(ante_pool / players_count) if players_count else 0)
    )
    _sb_nick, sb_shown = _blind_owner(reading, _SB_LABELS)
    _bb_nick, bb_shown = _blind_owner(reading, _BB_LABELS)
    blind_unit = next(
        (p.unit for p in reading.blinds_block if p.label.strip().casefold() in _BB_LABELS),
        reading.blind_unit,
    )
    sb_chips = _chips(sb_shown if sb_shown is not None else reading.sb, blind_unit, bb_chips)
    bb_amount = _chips(bb_shown if bb_shown is not None else reading.bb, blind_unit, bb_chips)
    bb_chips_final = bb_amount or bb_chips

    actions = [
        RawAction(
            street=action.street,
            label=labels.get(action.nickname, HERO_LABEL),
            kind=action.kind,
            amount=(
                _chips(action.amount, reading.pot_unit or Unit.BB, bb_chips)
                if action.amount is not None
                else None
            ),
            to_amount=(
                _chips(action.to_amount, reading.pot_unit or Unit.BB, bb_chips)
                if action.to_amount is not None
                else None
            ),
            is_all_in=action.is_all_in,
            raw_line=f"{action.position or ''} {action.kind.value}".strip(),
        )
        for action in reading.actions
    ]

    posts = [Post(label=label, kind=PostKind.ANTE, amount=ante_each) for label in ordered_labels]
    if sb_chips:
        posts.append(Post(label=labels[ring[0]], kind=PostKind.SMALL_BLIND, amount=sb_chips))
    if bb_amount and players_count > 1:
        posts.append(Post(label=labels[ring[1]], kind=PostKind.BIG_BLIND, amount=bb_amount))

    commits = _street_commits(actions, posts)
    total_commit = {
        label: sum(street.get(label, 0) for street in commits.values()) for label in labels.values()
    }
    uncalled: list[Uncalled] = []
    returned: dict[str, int] = {}
    for street_commits in commits.values():
        label, amount = _uncalled_by_player(street_commits)
        if label is not None and amount > 0:
            uncalled.append(Uncalled(label=label, amount=amount))
            returned[label] = returned.get(label, 0) + amount

    seats: list[SeatInfo] = []
    truncation_bump = 0
    for index, nickname in enumerate(ring, start=1):
        label = labels[nickname]
        player = by_nick.get(nickname)
        shown = _chips(
            player.stack if player is not None else None,
            (player.stack_unit if player is not None else None) or Unit.BB,
            bb_chips,
        )
        start = shown + ante_each + total_commit.get(label, 0) - returned.get(label, 0)
        floor = ante_each + total_commit.get(label, 0)
        if start < floor:
            truncation_bump = max(truncation_bump, floor - start)
            start = floor
        seats.append(SeatInfo(seat=index, label=label, stack=start))
    bump_bb = truncation_bump / (bb_amount or bb_chips)
    checks.append(
        VisionCheck(
            name=CHECK_TRUNCATION,
            passed=bump_bb <= _TRUNCATION_LIMIT_BB,
            detail=(
                f"стек пришлось поднять до вложенного на {bump_bb:.3f} ББ "
                f"(предел обрезки экрана {_TRUNCATION_LIMIT_BB} ББ)"
            ),
        )
    )

    hero_player = by_nick.get(hero_nickname) if hero_nickname is not None else None
    complete = reading.result_seen or reading.showdown_seen or bool(reading.winners)
    visible_bets = {
        labels[p.nickname]: _chips(p.bet, p.bet_unit or Unit.BB, bb_chips)
        for p in reading.players
        if p.bet and p.nickname in labels
    }
    collected = [
        Collected(
            label=labels.get(win.nickname, HERO_LABEL),
            amount=_chips(win.amount, win.unit or Unit.BB, bb_chips),
        )
        for win in reading.winners
        if win.amount
    ]

    raw = RawHand(
        provenance=Provenance.SCREENSHOT,
        completeness=Completeness.HAND if complete else Completeness.STATE,
        source_ref=source_ref,
        hand_no=reading.hand_no or "",
        tournament_id="",
        tournament_name=reading.tournament_name or "",
        level=reading.level or 0,
        sb=sb_chips,
        bb=bb_chips_final,
        ante=ante_each,
        timestamp=_NO_TIMESTAMP,
        table_name="",
        max_seats=reading.max_seats or players_count,
        button_seat=len(seats) if len(seats) > 2 else 1,
        seats=seats,
        visible_bets={} if complete else visible_bets,
        posts=posts,
        dealt=(
            {HERO_LABEL: _cards(hero_player)}
            if hero_player is not None and _cards(hero_player)
            else {}
        ),
        actions=actions if complete else _folds_of_absent_cards(reading, labels),
        boards=_boards(reading.board if _board_was_dealt(reading) else []),
        uncalled=uncalled,
        showdowns=_showdowns(reading, labels) if complete else [],
        collected=collected,
        vision=VisionMeta(
            image_hash=image_hash,
            nicknames={
                labels[p.nickname]: p.nickname
                for p in reading.players
                if p.nickname is not None and p.nickname in labels
            },
            # Баунти — в центах: контракт хранит целое, а экран печатает доллары
            # с двумя знаками. Из EV они не участвуют никак (решение владельца
            # 2026-09-05), и доезжают сюда ради заметок и правила по знаку.
            bounties={
                labels[p.nickname]: round((p.bounty_usd or 0) * 100)
                for p in reading.players
                if p.bounty_usd and p.nickname in labels
            },
            displayed_pot=_chips(reading.pot_shown, reading.pot_unit, bb_chips),
            unsure_fields=list(reading.unsure_fields),
            hero_candidates=[p.nickname for p in reading.players if p.nickname],
        ),
    )
    return raw, checks


def _folds_of_absent_cards(
    reading: VisionReading, labels: dict[str | None, str]
) -> list[RawAction]:
    """Пас у каждого, перед кем на живом столе не видно карт.

    Единственный видимый признак «не в руке» на живом столе — отсутствие карт
    перед игроком; пас и есть «не в руке», никакой суммы и никакого намерения он
    не несёт.
    """
    return [
        RawAction(
            street=Street.PREFLOP,
            label=labels[p.nickname],
            kind=ActionKind.FOLD,
            raw_line="перед игроком нет карт",
        )
        for p in reading.players
        if p.has_hole_cards is False and p.nickname in labels
    ]


def _board_was_dealt(reading: VisionReading) -> bool:
    """Дошла ли рука до карт стола вообще — по правилам покера, а не по яркости.

    Экспорт дорисовывает борд приглушённым даже там, где рука кончилась до
    флопа: последний рейз никто не заколлировал, банк ушёл сразу. Отличить такие
    карты глазом модель может (`board_faded`), но полагаться на одну яркость
    нельзя — см. отчёт прогона датасета.

    Кодовое правило независимо от яркости: карты стола раздают, только если
    торговля пошла дальше префлопа ЛИБО дело дошло до вскрытия. Ни того ни
    другого — борда в руке не было, чем бы экран его ни рисовал.

    Вскрытие считается и по флагу `showdown_seen`, и по прочитанному: карты
    видны у двоих и более — значит, до вскрытия дошло, даже если флаг забыт.
    Ошибиться этим правилом дешевле в сторону «раздан»: раздан ли борд, при
    сомнении решает валидатор — недостающие карты стола он называет, лишние нет.
    """
    return (
        reading.showdown_seen
        or any(action.street is not Street.PREFLOP for action in reading.actions)
        or sum(1 for player in reading.players if _cards(player)) >= 2
    )


def _boards(board: list[str]) -> dict[Street, list[str]]:
    """Карты стола, разложенные по улицам: три флопа, одна тёрна, одна ривера."""
    split = _BOARD_SPLIT.get(len(board))
    if split is None:
        return {}
    boards: dict[Street, list[str]] = {}
    offset = 0
    for street, count in split.items():
        boards[street] = board[offset : offset + count]
        offset += count
    return boards


def _showdowns(
    reading: VisionReading, labels: dict[str | None, str]
) -> list[ShowdownEntry]:
    return [
        ShowdownEntry(label=labels[p.nickname], cards=_cards(p))
        for p in reading.players
        if len(_cards(p)) == 2 and p.nickname in labels
    ]


# --- контрольные суммы и каскад ---------------------------------------------


def contributions_bb(raw: RawHand) -> float:
    """Сумма видимых вкладов в тех же ББ, в которых напечатан банк.

    Считается по УЖЕ построенной руке, а не по чтению: вклады там сведены к одним
    единицам, а возвращённое непоколленное вычтено — банк на экране показан после
    возврата. Публичная: по ней же проверяется ответ игрока про банк
    (`apply_vision_answer`) — считать эту сумму двумя формулами нельзя, разойдясь,
    они дали бы «сошлось» на одном пути и «не сошлось» на другом.
    """
    total = raw.ante * len(raw.seats)
    commits = _street_commits(raw.actions, raw.posts)
    total += sum(sum(street.values()) for street in commits.values())
    total -= sum(entry.amount for entry in raw.uncalled)
    return total / raw.bb if raw.bb else 0.0


# Борд, открытый НА МОМЕНТ олл-ина, по улице последнего действия: GG печатает
# проценты именно для этого момента, а не для законченной руки. На законченной
# доске эквити равно нулю или единице, и сверка с ней ловила бы каждую руку.
_BOARD_AT_ALL_IN: dict[Street, tuple[Street, ...]] = {
    Street.PREFLOP: (),
    Street.FLOP: (Street.FLOP,),
    Street.TURN: (Street.FLOP, Street.TURN),
    Street.RIVER: (Street.FLOP, Street.TURN, Street.RIVER),
}


def _board_at_all_in(raw: RawHand) -> list[str]:
    """Карты стола, лежавшие на момент последнего действия руки."""
    street = raw.actions[-1].street if raw.actions else Street.PREFLOP
    return [card for s in _BOARD_AT_ALL_IN[street] for card in raw.boards.get(s, [])]


def _showdown_pair(raw: RawHand) -> tuple[list[str], list[str]]:
    """Две первые вскрытые руки — ЗАПАСНОЙ вход проверки эквити.

    Основной вход — игроки, чью долю подписал экран (`run_checks`): их бывает и
    трое. Пара из вскрытия остаётся для экранов, где процент напечатан, а карты
    читаются только из вскрытия, и там участников ровно двое.
    """
    hands = [entry.cards for entry in raw.showdowns if len(entry.cards) == 2]
    return (hands[0], hands[1]) if len(hands) >= 2 else ([], [])


def _printed_positions(reading: VisionReading) -> dict[str, str]:
    """Метки позиций, НАПЕЧАТАННЫЕ у строк лога, по нику — как прочитано."""
    printed: dict[str, str] = {}
    for action in reading.actions:
        if action.nickname and action.position:
            printed.setdefault(action.nickname, action.position.strip().upper())
    return printed


# Как GG подписывает позиции в логе против того, как их называет нормалайзер.
# Эти три соответствия от размера стола не зависят: блайнды и кнопка есть в
# любом круге.
_GG_POSITION_ALIASES: dict[str, str] = {"ББ": "BB", "БТН": "BTN", "МБ": "SB"}

# Середина стола подписана у рума иначе, и наблюдалось это только на 8-max
# экспорте (реестр A3): там `MP`/`MP+1` стоят на местах, которые нормалайзер
# зовёт `LJ`/`HJ`. На других размерах соответствие не измерено, поэтому там
# метка остаётся неопознанной — и сверка её просто не сравнивает, а не роняет
# (`vision_checks.positions_check`).
_GG_POSITION_ALIASES_8MAX: dict[str, str] = {"MP": "LJ", "MP+1": "HJ"}


def _aliased_position(label: str, seats: int) -> str:
    """Метка рума в словаре нормалайзера — насколько соответствие измерено."""
    if seats == 8 and label in _GG_POSITION_ALIASES_8MAX:
        return _GG_POSITION_ALIASES_8MAX[label]
    return _GG_POSITION_ALIASES.get(label, label)


def _derived_positions(raw: RawHand) -> dict[str, str]:
    """Позиции, ВОССТАНОВЛЕННЫЕ по кругу мест, по нику — вход сверки с печатью."""
    from harness.normalizer import POSITIONS_BY_COUNT

    order = POSITIONS_BY_COUNT.get(len(raw.seats))
    if order is None or raw.vision is None:
        return {}
    by_label = dict(zip([seat.label for seat in raw.seats], order, strict=True))
    return {
        nickname: by_label[label]
        for label, nickname in raw.vision.nicknames.items()
        if label in by_label
    }


def run_checks(
    reading: VisionReading, raw: RawHand, hero_check: VisionCheck, built: list[VisionCheck]
) -> list[VisionCheck]:
    """Все контрольные суммы по одному чтению — в порядке их доказательной силы."""
    at_seat = {p.nickname or "": p.cards_at_seat for p in reading.players if p.cards_at_seat}
    in_log = {p.nickname or "": p.cards_in_log for p in reading.players if p.cards_in_log}
    hero_cards, villain_cards = _showdown_pair(raw)
    # Участники олл-ина для оракула эквити — ВСЕ, чью долю экран подписал, а не
    # первый из них: GG печатает долю каждого, и посчитанная на двоих доля
    # трёхстороннего олл-ина расходится с экраном на десяток процентных единиц.
    # Пара из вскрытия остаётся запасным входом для экранов, где процент
    # напечатан, а карты читаются только из вскрытия.
    equity_hands: list[tuple[list[str], float | None]] = [
        (_cards(p), p.equity_shown_pct)
        for p in reading.players
        if p.equity_shown_pct is not None and len(_cards(p)) == 2
    ]
    if len(equity_hands) < 2:
        shown_pct = next(
            (p.equity_shown_pct for p in reading.players if p.equity_shown_pct is not None), None
        )
        equity_hero = equity_hands[0][0] if equity_hands else hero_cards
        other = villain_cards if equity_hero == hero_cards else hero_cards
        equity_hands = [(equity_hero, shown_pct), (other, None)]
    printed = {
        nick: _aliased_position(pos, len(raw.seats))
        for nick, pos in _printed_positions(reading).items()
    }
    return [
        hero_check,
        *built,
        seats_check(len(reading.players), reading.max_seats),
        positions_check(printed, _derived_positions(raw)),
        cards_check(at_seat, in_log),
        pot_check(
            (reading.pot_shown if reading.pot_unit is not Unit.CHIPS else None),
            contributions_bb(raw),
        ),
        equity_check(equity_hands, _board_at_all_in(raw)),
    ]


def _seat_of_nickname(raw: RawHand, nickname: str) -> str | None:
    """Метка места по нику — обратный ход к `VisionMeta.nicknames`."""
    if raw.vision is None:
        return None
    for label, nick in raw.vision.nicknames.items():
        if nick == nickname:
            return label
    return None


# Поля эскалации, ответ на которые код умеет подставить в руку. Вопрос по полю
# вне этого списка — мёртвый: игрок отвечает, ответ ложится в eval-датасет, а
# рука остаётся прежней, и следующий проход упирается в то же расхождение. Такие
# поля спрашивать нельзя вовсе (ревью раунда 1, R3).
ANSWERABLE_FIELDS = frozenset({"pot", "button", "hero", "cards"})


def can_apply_vision_answer(field: str) -> bool:
    """Умеет ли код подставить ответ игрока по этому полю в сырую руку."""
    return field in ANSWERABLE_FIELDS


def _resolved(raw: RawHand, field: str) -> VisionMeta:
    """Пометить проверку `field` закрытой ответом игрока.

    Ответ игрока и есть разрешение спора для своего поля: он видел экран, а мы
    нет. Пометка нужна не для красоты — по ней станция отличает «расхождение
    закрыто» от «спросили и не помогло», и второе больше не доходит до вердикта
    (ревью раунда 1, R1).
    """
    meta = raw.vision or VisionMeta()
    return meta.model_copy(
        update={
            "checks": [
                check.model_copy(update={"passed": True, "detail": "закрыто ответом игрока"})
                if check.name == field and not check.passed
                else check
                for check in meta.checks
            ]
        }
    )


def _cards_of_answer(value: str) -> list[str]:
    """Две карты из строки варианта («Ks Ad»). Иначе пусто — подставлять нечего."""
    cards = value.split()
    return cards if len(cards) == 2 else []


def apply_vision_answer(
    raw: RawHand, field: str, value: str, *, subject: str = ""
) -> RawHand | None:
    """Подставить ответ игрока в сырую руку — спека §8.3, шаг 2.

    `None` означает «этим ответом руку не поправить». Ответ при этом уже записан
    в `eval_cases` вызывающим и не теряется: он размеченный пример независимо от
    того, помог ли он этой конкретной руке. Но до вердикта такая рука не
    доходит — станция видит непройденную проверку и отказывается (см.
    `worker.pipeline._unresolved_checks`).

    Патчатся ровно те поля, у которых ответ игрока однозначно ложится в контракт
    (`ANSWERABLE_FIELDS`):

    * `pot` — показанный банк (`VisionMeta.displayed_pot`);
    * `button` — кнопка переставляется на место названного игрока;
    * `hero` — герой переименовывается в названного игрока;
    * `cards` — карты названного игрока (`subject` — его ник) ставятся и в
      раздачу, и во вскрытие: расхождение было между двумя прочтениями ОДНИХ
      карт, и разводить их после ответа не во что.

    Ни одно из значений не «подгоняется, чтобы сошлось»: подставляется ровно то,
    что сказал игрок, а сойдётся ли после этого рука, решает валидатор на
    следующем проходе.
    """
    if field == "pot":
        try:
            shown = float(value.replace(",", "."))
        except ValueError:
            return None
        # На ПОЛНОЙ руке `displayed_pot` не читает никто: банк там считает движок
        # по вкладам, и записать ответ игрока в это поле — значит не изменить
        # ничего. Поэтому ответ засчитывается за разрешение спора, только если он
        # с этими вкладами и СХОДИТСЯ; иначе поле пишется (ответ игрока не
        # теряется), но проверка остаётся непройденной, и станция до вердикта не
        # доходит. Иначе получалось так: модель пропустила анте, игрок подтвердил
        # показанный банк, проверка «закрылась» — и разбор уезжал игроку по руке
        # с анте, равным нулю (ревью раунда 2, F1).
        agrees = within_tolerance(abs(shown - contributions_bb(raw)), POT_TOLERANCE_BB)
        meta = _resolved(raw, field) if agrees else raw.vision or VisionMeta()
        return raw.model_copy(
            update={"vision": meta.model_copy(update={"displayed_pot": round(shown * raw.bb)})}
        )

    if field == "cards":
        cards = _cards_of_answer(value)
        label = _seat_of_nickname(raw, subject) if subject else None
        if not cards or label is None:
            return None
        showdowns = [
            entry.model_copy(update={"cards": cards}) if entry.label == label else entry
            for entry in raw.showdowns
        ]
        if all(entry.label != label for entry in raw.showdowns):
            showdowns = [*showdowns, ShowdownEntry(label=label, cards=cards)]
        dealt = {**raw.dealt}
        if label in dealt or label == HERO_LABEL:
            dealt[label] = cards
        return raw.model_copy(
            update={"dealt": dealt, "showdowns": showdowns, "vision": _resolved(raw, field)}
        )

    if field == "button":
        label = _seat_of_nickname(raw, value)
        seat = next((s.seat for s in raw.seats if s.label == label), None)
        if seat is None:
            return None
        return raw.model_copy(update={"button_seat": seat, "vision": _resolved(raw, field)})

    if field == "hero":
        label = _seat_of_nickname(raw, value)
        if label is None or label == HERO_LABEL:
            return None
        renamed = {
            HERO_LABEL: label,
            label: HERO_LABEL,
        }
        seats = [
            s.model_copy(update={"label": renamed.get(s.label, s.label)}) for s in raw.seats
        ]
        posts = [
            p.model_copy(update={"label": renamed.get(p.label, p.label)}) for p in raw.posts
        ]
        actions = [
            a.model_copy(update={"label": renamed.get(a.label, a.label)}) for a in raw.actions
        ]
        meta = _resolved(raw, field).model_copy(
            update={
                "nicknames": {
                    renamed.get(lbl, lbl): nick
                    for lbl, nick in (raw.vision.nicknames if raw.vision else {}).items()
                }
            }
        )
        return raw.model_copy(
            update={"seats": seats, "posts": posts, "actions": actions, "vision": meta}
        )

    return None


async def vision_extract(
    llm: VisionLLM,
    image: bytes,
    *,
    gg_nickname: str,
    trace_id: int,
    source_ref: str = "screenshot",
    image_hash: str | None = None,
    fallback_available: bool = True,
    prompt_path: Path = _PROMPT_PATH,
) -> VisionOutcome:
    """Прочитать экран и вернуть руку — с каскадом моделей и контрольными суммами.

    Ступени: `LLM_VISION_MODEL`; при провале любой проверки — `LLM_VISION_FALLBACK
    _MODEL`; при провале и там — эскалация игроку. Дорогая ступень пропускается,
    когда её модель не настроена (`fallback_available=False`): переменная
    окружения необязательна, и её отсутствие означает «каскада нет», а не отказ.

    Отказ модели (`not_a_hand`) каскад НЕ запускает: честный отказ — это ответ, а
    не сбой, и платить за его повторение второй раз незачем (реестр C3).

    **Незавершённая рука обрывает каскад там же, на первой ступени.** Решение
    владельца 2026-09-09: экран, на котором не видно, чем раздача кончилась
    (`Completeness.STATE`), не разбирается вовсе. Дорогая ступень читала бы тот
    же экран, а вопрос игроку уточнял бы числа руки, которую мы всё равно
    откажемся разбирать, — обе траты не окупаются ничем
    (`test_a_hand_in_progress_stops_the_cascade_on_the_first_hop`).
    """
    prompt = read_prompt(prompt_path)
    hops: list[VisionHop] = []
    roles: list[tuple[str, Literal["vision_extract", "vision_extract_fallback"]]] = [
        ("primary", "vision_extract")
    ]
    if fallback_available:
        roles.append(("fallback", "vision_extract_fallback"))

    outcome: VisionOutcome | None = None
    for role, purpose in roles:
        reading, meta = await llm(
            purpose, VisionReading, prompt=prompt, images=[image], trace_id=trace_id
        )
        model = getattr(meta, "model", purpose)
        if _is_empty(reading):
            # Пустое чтение — сбой, а не ответ, и лечится оно повтором.
            hops.append(VisionHop(role=role, model=model, error=_EMPTY_READING))
            reading, meta = await llm(
                purpose, VisionReading, prompt=prompt, images=[image], trace_id=trace_id
            )
            model = getattr(meta, "model", purpose)
            if _is_empty(reading):
                hops.append(VisionHop(role=role, model=model, error=_EMPTY_READING))
                continue
        if reading.not_a_hand:
            hops.append(VisionHop(role=role, model=model, error="not_a_hand"))
            return VisionOutcome(
                raw=None,
                hops=hops,
                refusal=reading.refusal_reason or "экран не похож на покерную раздачу",
            )
        hero_nickname, hero_check = match_hero(
            gg_nickname, [p.nickname for p in reading.players if p.nickname]
        )
        raw, built = reading_to_raw(
            reading,
            hero_nickname=hero_nickname,
            source_ref=source_ref,
            image_hash=image_hash,
        )
        if raw.completeness is Completeness.STATE:
            hops.append(VisionHop(role=role, model=model, error=_HAND_IN_PROGRESS))
            return VisionOutcome(raw=None, hops=hops, hand_in_progress=True)
        checks = run_checks(reading, raw, hero_check, built)
        failed = [check.name for check in checks if not check.passed]
        hops.append(VisionHop(role=role, model=model, failed_checks=failed))
        raw.vision = (raw.vision or VisionMeta()).model_copy(
            update={"hops": list(hops), "checks": checks}
        )
        outcome = VisionOutcome(raw=raw, checks=checks, hops=list(hops))
        if not failed:
            return outcome
    if outcome is None:
        raise VisionReadFailed(
            "модель вернула пустое чтение на каждой ступени каскада — экран не прочитан"
        )
    return outcome
