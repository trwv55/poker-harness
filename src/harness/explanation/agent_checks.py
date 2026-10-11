"""Проверки ответа агента и сводка разбора для его контекста (спека агента №1, §4, §6).

Одна реализация на прод (`harness.agent`) и evals (спека №2): вторая разошлась
бы с первой на первой же правке. Модуль чистый — ни pydantic-ai, ни базы:
разрешённые числа, метки и признак вызова `leaks` приходят аргументами.

**Граница, а не гарантия.** Допущение, термины и лучшая линия проверяются по
явным оборотам; что сказано другими словами, ловит судья (спека №2).

**Обозначения карт — не числа.** `A5s`, `T9o`, `KQ`, карта с мастью, ранги борда
вырезаются ДО сверки чисел. Чисто цифровая пара (`99`) от суммы шаблоном не
отличима (тех-спека §5.7); решение — она карта, только если это класс руки
героя в сводке (`test_a_pocket_pair_is_a_card_only_when_the_hero_holds_it`).
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
from harness.explanation.faithfulness import error_words_in, numbers_in, unsupported_numbers
from harness.explanation.hand_replay import hand_replay
from harness.explanation.verdict_text import verdict_digest

__all__ = [
    "ASSUMPTION_MARK",
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
_BOARD_RE = re.compile(_EDGE_L + _RANK + r"(?:[-–]?" + _RANK + r"){2,4}" + _EDGE_R)
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

# Термин → метки, при которых он допустим: значения `ActionTag`, `HandCategory`,
# `DrawKind` (контекст разбора) и `FrequencyStat` (выход расчёта).
TERM_LABELS: dict[str, tuple[re.Pattern[str], frozenset[str]]] = {
    "c-бет": (
        re.compile(r"[cс]-?бет|контбет", re.IGNORECASE),
        frozenset({"cbet", "delayed_cbet", "cbet_flop", "fold_to_cbet"}),
    ),
    "донк": (re.compile(r"(?<![а-яё])донк", re.IGNORECASE), frozenset({"donk"})),
    "проба": (
        re.compile(r"(?<![а-яё])проб(?:а|у|ой|ы|е)(?![а-яё])", re.IGNORECASE),
        frozenset({"probe"}),
    ),
    "баррель": (
        re.compile(r"баррел", re.IGNORECASE),
        frozenset({"barrel", "barrel_turn", "barrel_river"}),
    ),
    "чек-рейз": (re.compile(r"чек-?рейз", re.IGNORECASE), frozenset({"check_raise"})),
    "сет": (
        re.compile(r"(?<![а-яё])сет(?:а|ом|у|ы)?(?![а-яё])", re.IGNORECASE),
        frozenset({"set"}),
    ),
    "оверпара": (re.compile(r"оверпар", re.IGNORECASE), frozenset({"overpair"})),
    "топ-пара": (
        re.compile(r"топ-?пар", re.IGNORECASE),
        frozenset({"top_pair_strong_kicker", "top_pair_weak_kicker"}),
    ),
    "гатшот": (re.compile(r"гатшот", re.IGNORECASE), frozenset({"gutshot", "double_gutshot"})),
    "флеш-дро": (re.compile(r"флеш-?дро", re.IGNORECASE), frozenset({"flush"})),
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
    метки классификатора всех точек. Каждая величина — через `NumberBook`;
    идентификаторы в заголовке точки — нет, как в выжимке вердикта
    (`test_the_context_registers_every_number_it_prints`)."""
    book = NumberBook()
    replay = book.prose(hand_replay(enriched).plain)
    verdict = verdict_digest(result)
    labels: set[str] = set()
    label_lines: list[str] = []
    for point in result.points:
        line = postflop_line_detail(point)
        if line is None:
            continue
        found = []
        if line.hand is not None:
            found.append(line.hand.category.value)
        if line.draw is not None:
            found.extend(kind.value for kind in line.draw.kinds)
        if line.line is not None:
            found.append(line.line.action.value)
        labels.update(found)
        label_lines.append(
            f"Точка {point.street.value}: метки классификатора — {', '.join(found)}."
        )
    hand = enriched.hand
    cards = hand.dealt.get(hand.hero_label, [])
    hero_class = class_of(cards[0], cards[1]) if len(cards) == 2 else None
    # Порядок объявления `Street`, как у реплея: `sorted` по строке дал бы
    # «flop, river, turn».
    board = tuple(card[0] for street in Street for card in hand.boards.get(street, []))
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
        ranks = re.findall(_RANK, match.group(0))
        has_letter = any(rank in "TJQKA" for rank in ranks)
        if (board and not (Counter(ranks) - board)) or (has_letter and not board):
            return " "
        return match.group(0)

    def class_sub(match: re.Match[str]) -> str:
        high, low, suit = match.group(1), match.group(2), match.group(3)
        if high.isdigit() and low.isdigit() and suit is None:
            return " " if match.group(0) == hero_class else match.group(0)
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
            numbers = numbers_in(cleaned(sentence))
            if not numbers or unsupported_numbers(cleaned(sentence), allowed):
                violations.append(
                    f"утверждение о диапазоне без метки {ASSUMPTION_MARK} — «{quoted}»"
                )
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
