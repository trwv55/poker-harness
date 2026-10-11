"""Проверки ответа агента и сводка разбора для его контекста (спека агента №1, §4, §6).

Одна реализация на прод (`harness.agent`) и evals (спека №2): вторая разошлась
бы с первой на первой же правке. Модуль чистый — ни pydantic-ai, ни базы:
разрешённые числа, метки и признак вызова `leaks` приходят аргументами.

**Диапазон — всегда с меткой.** В v1 диапазоны не считает ни один расчёт
(они — спека №3), поэтому любое предложение о диапазоне без
«(допущение)» отвергается, даже если его числа разрешены: приписка любого факта
руки иначе снимала бы метку (`test_a_range_sentence_with_allowed_numbers_still_needs_the_mark`).
Исключение для чисел из расчёта диапазонов может вернуться в спеке №3.

**Граница, а не гарантия.** Допущение, термины и лучшая линия проверяются по
явным оборотам; что сказано другими словами, ловит судья (спека №2).

**Обозначения карт — не числа.** `A5s`, `T9o`, `KQ`, карта с мастью, ранги борда
вырезаются ДО сверки чисел. Чисто цифровая пара (`99`) от суммы шаблоном не
отличима (тех-спека §5.7); решение — она карта, только если это класс руки
героя в сводке (`test_a_pocket_pair_is_a_card_only_when_the_hero_holds_it`) и
сразу за ней не стоит единица суммы («99 BB» — число,
`test_a_hero_pair_followed_by_a_unit_stays_a_number`). Цифры борда без
разделителя («258») — сумма, а не борд (`test_digits_without_separators_are_not_a_board`).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from harness.contracts import (
    AnalysisResult,
    EnrichedHand,
    Street,
    class_of,
    postflop_line_detail,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import error_words_in, unsupported_numbers
from harness.explanation.hand_replay import hand_replay
from harness.explanation.verdict_text import verdict_digest

__all__ = [
    "ACTION_LABEL",
    "ASSUMPTION_MARK",
    "DRAW_LABEL",
    "HAND_LABEL",
    "STAT_LABEL",
    "TERM_LABELS",
    "AgentContext",
    "agent_context",
    "check_agent_answer",
    "strip_card_tokens",
]

ASSUMPTION_MARK = "(допущение)"

_RANK = r"(?:10|[2-9TJQKA])"
_EDGE_L = r"(?<!\w)(?<!\d[.,])"
_EDGE_R = r"(?![\w%]|[.,]\d)"
_SUITED_CARD_RE = re.compile(_RANK + r"[♠♥♦♣]️?")
_CLASS_RE = re.compile(_EDGE_L + r"([2-9TJQKA])([2-9TJQKA])([so])?" + _EDGE_R)
# Без разделителя ранги борда слипаются, только если рядом буква («KQ7»): три
# цифры подряд («258») — сумма, а не борд 2-5-8.
_BOARD_JOIN = r"(?:[-–]|(?<=[TJQKA])|(?=[TJQKA]))"
_BOARD_RE = re.compile(_EDGE_L + _RANK + r"(?:" + _BOARD_JOIN + _RANK + r"){2,4}" + _EDGE_R)
# Единица сразу после числа делает цифровую пару суммой, даже если герой держит
# эту пару: «99 BB», «стек 22 BB», «22 раздачи».
_AMOUNT_AFTER_RE = re.compile(r"\s*(?:bb|бб|%|фиш|раздач|рук|процент|из\b)", re.IGNORECASE)
# Точка внутри числа («6.6») предложение не кончает.
_SENTENCE_RE = re.compile(r"(?:[^.!?\n]|\.(?=\d))+[.!?]?")
_RANGE_RE = re.compile(r"диапазон|рендж", re.IGNORECASE)
# «стоило» — только с инфинитивом или «бы»: «ничего не стоило» — не упрёк.
_BEST_LINE_RE = re.compile(
    r"лучше было|было лучше|лучше бы|правильн\w* было бы|следовало|надо было|нужно было"
    r"|стоило\s+(?:бы|[а-яё]+ть)\b",
    re.IGNORECASE,
)
_FLOP_TURN_RE = re.compile(r"флоп|т[её]рн", re.IGNORECASE)
_LEAK_RE = re.compile(r"(?<![а-яё])лик(?:и|ов|ам|ами|ах|а|ом|у|е)?(?![а-яё])", re.IGNORECASE)

# Метки — с пространством имён: `HandCategory.FLUSH` и `DrawKind.FLUSH` — одна
# строка `"flush"`, и без префикса «флеш-дро» прошёл бы на готовом флеше.
# `hand:` — `HandCategory`, `draw:` — `DrawKind`, `action:` — `ActionTag`
# (контекст разбора, `agent_context`); `stat:` — `FrequencyStat` (выход расчёта,
# метку собирает вызывающий).
HAND_LABEL = "hand:"
DRAW_LABEL = "draw:"
ACTION_LABEL = "action:"
STAT_LABEL = "stat:"

# Термин → метки, при которых он допустим.
TERM_LABELS: dict[str, tuple[re.Pattern[str], frozenset[str]]] = {
    "c-бет": (
        re.compile(r"[cс]-?бет|контбет", re.IGNORECASE),
        frozenset({"action:cbet", "action:delayed_cbet", "stat:cbet_flop", "stat:fold_to_cbet"}),
    ),
    "донк": (re.compile(r"(?<![а-яё])донк", re.IGNORECASE), frozenset({"action:donk"})),
    "проба": (
        re.compile(r"(?<![а-яё])проб(?:а|у|ой|ы|е)(?![а-яё])", re.IGNORECASE),
        frozenset({"action:probe"}),
    ),
    "баррель": (
        re.compile(r"баррел", re.IGNORECASE),
        frozenset({"action:barrel", "stat:barrel_turn", "stat:barrel_river"}),
    ),
    "чек-рейз": (re.compile(r"чек-?рейз", re.IGNORECASE), frozenset({"action:check_raise"})),
    "сет": (
        re.compile(r"(?<![а-яё])сет(?:а|ом|у|ы)?(?![а-яё])", re.IGNORECASE),
        frozenset({"hand:set"}),
    ),
    "оверпара": (re.compile(r"оверпар", re.IGNORECASE), frozenset({"hand:overpair"})),
    "топ-пара": (
        re.compile(r"топ-?пар", re.IGNORECASE),
        frozenset({"hand:top_pair_strong_kicker", "hand:top_pair_weak_kicker"}),
    ),
    "гатшот": (
        re.compile(r"гатшот", re.IGNORECASE),
        frozenset({"draw:gutshot", "draw:double_gutshot"}),
    ),
    "флеш-дро": (re.compile(r"флеш-?дро", re.IGNORECASE), frozenset({"draw:flush"})),
}


_STREET_NAME: dict[Street, str] = {
    Street.PREFLOP: "префлоп",
    Street.FLOP: "флоп",
    Street.TURN: "тёрн",
    Street.RIVER: "ривер",
}


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Сводка последнего разбора для агента: текст, разрешённые числа, метки
    классификатора, класс руки героя и ранги борда (для вырезания карт)."""

    text: str
    allowed: frozenset[float]
    labels: frozenset[str]
    hero_class: str | None
    board_ranks: tuple[str, ...]


def agent_context(result: AnalysisResult, enriched: EnrichedHand) -> AgentContext:
    """Сводка разбора: ход руки (реплей, код), судимые точки (выжимка вердикта),
    метки классификатора всех точек. Разрешены числа реплея без обозначений
    карт: ранг карты героя или борда числом не становится
    (`test_a_card_rank_of_the_hand_is_not_an_allowed_number`). Идентификаторы в
    заголовке точки выжимки не разрешены, как в выжимке вердикта
    (`test_the_context_registers_every_number_it_prints`)."""
    hand = enriched.hand
    cards = hand.dealt.get(hand.hero_label, [])
    hero_class = class_of(cards[0], cards[1]) if len(cards) == 2 else None
    # Порядок объявления `Street`, как у реплея: `sorted` по строке дал бы
    # «flop, river, turn».
    board = tuple(card[0] for street in Street for card in hand.boards.get(street, []))
    book = NumberBook()
    replay = hand_replay(enriched).plain
    # Регистрируется копия без карт; модели показывается реплей как есть.
    book.prose(strip_card_tokens(replay, hero_class=hero_class, board_ranks=board))
    verdict = verdict_digest(result)
    labels: set[str] = set()
    label_lines: list[str] = []
    for point in result.points:
        line = postflop_line_detail(point)
        if line is None:
            continue
        found = []
        if line.hand is not None:
            found.append(HAND_LABEL + line.hand.category.value)
        if line.draw is not None:
            found.extend(DRAW_LABEL + kind.value for kind in line.draw.kinds)
        if line.line is not None:
            found.append(ACTION_LABEL + line.line.action.value)
        labels.update(found)
        label_lines.append(
            f"Метки классификатора, {_STREET_NAME[point.street]}: {', '.join(found)}."
        )
    text = "\n\n".join(part for part in (replay, verdict.text, "\n".join(label_lines)) if part)
    return AgentContext(
        text=text,
        allowed=book.allowed | verdict.allowed,
        labels=frozenset(labels),
        hero_class=hero_class,
        board_ranks=board,
    )


def strip_card_tokens(
    text: str, *, hero_class: str | None = None, board_ranks: Sequence[str] = ()
) -> str:
    """Текст без обозначений карт — то, что идёт в сверку чисел."""
    text = _SUITED_CARD_RE.sub(" ", text)
    board = Counter(board_ranks)

    def board_sub(match: re.Match[str]) -> str:
        ranks = ["T" if rank == "10" else rank for rank in re.findall(_RANK, match.group(0))]
        has_letter = any(rank in "TJQKA" for rank in ranks)
        if (board and not (Counter(ranks) - board)) or (has_letter and not board):
            return " "
        return match.group(0)

    def class_sub(match: re.Match[str]) -> str:
        high, low, suit = match.group(1), match.group(2), match.group(3)
        if high.isdigit() and low.isdigit() and suit is None:
            amount = _AMOUNT_AFTER_RE.match(match.string, match.end()) is not None
            return " " if match.group(0) == hero_class and not amount else match.group(0)
        return " "

    return _CLASS_RE.sub(class_sub, _BOARD_RE.sub(board_sub, text))


def check_agent_answer(
    answer: str,
    *,
    allowed: frozenset[float],
    labels: frozenset[str],
    leaks_called: bool,
    hero_class: str | None = None,
    board_ranks: Sequence[str] = (),
) -> list[str]:
    """Нарушения ответа словами; пустой список — ответ прошёл (спека §6)."""
    text = answer.strip()
    if not text:
        return ["пустой ответ"]

    def cleaned(part: str) -> str:
        return strip_card_tokens(part, hero_class=hero_class, board_ranks=board_ranks)

    violations: list[str] = []
    invented = unsupported_numbers(cleaned(text), allowed)
    if invented:
        violations.append(f"числа не из расчёта и не из разбора — {invented}")
    for sentence in _SENTENCE_RE.findall(text):
        quoted = sentence.strip()
        if _RANGE_RE.search(sentence) and ASSUMPTION_MARK not in sentence:
            violations.append(f"утверждение о диапазоне без метки {ASSUMPTION_MARK} — «{quoted}»")
        if _BEST_LINE_RE.search(sentence) and _FLOP_TURN_RE.search(sentence):
            violations.append(f"лучшая линия на флопе или тёрне — «{quoted}»")
    for term, (pattern, needed) in TERM_LABELS.items():
        if pattern.search(text) and not (needed & labels):
            violations.append(f"термин «{term}» без такой метки классификатора")
    if _LEAK_RE.search(text) and not leaks_called:
        violations.append("слово «лик» без расчёта leaks")
    errors = error_words_in(text)
    if errors:
        violations.append(f"решение названо ошибкой — {errors}")
    return violations
