"""Проверки ответа агента и сводка разбора для его контекста (спека агента №1, §4, §6).

Одна реализация на прод (`harness.agent`) и evals (спека №2): вторая разошлась
бы с первой на первой же правке. Модуль чистый — ни pydantic-ai, ни базы:
разрешённые числа, метки и признак вызова `leaks` приходят аргументами.

**Допущение — с меткой.** В v1 диапазоны не считает ни один расчёт (они —
спека №3), поэтому предложение, которое говорит о сопернике (он, его, соперник,
поле, позиция, «такие игроки») и о том, что он держит или сделает (диапазон,
пары, `Jx`, сбрасывает, защитит, коллирует…), без «(допущение)» отвергается,
даже если его числа разрешены и даже с «если»
(`test_a_range_sentence_with_allowed_numbers_still_needs_the_mark`,
`test_a_claim_about_what_the_opponent_holds_or_does_needs_the_mark`). Ложный
отказ здесь сознательно дешевле ложного пропуска. Метки не требуют действие,
привязанное к доле или «N из M» из расчёта («сбрасывать 33.3% диапазона»,
«сбрасывало 30 из 50»), и метаречь («про его диапазон это мало говорит»)
(`test_calc_backed_statements_and_meta_speech_need_no_mark`).

**«В плюсе», «в минусе», «окупается» — вывод только по порогу.** Голый вывод
допустим, только когда в разговоре есть `frequency_vs_threshold` с выводом выше
или ниже порога (`test_a_profit_conclusion_needs_a_decided_threshold`); с долей и
условием («от 19.4% эквити», «при 24% фолдов», «если … чаще 24%») это
арифметика, и она допустима всегда (`test_profit_arithmetic_with_a_share_is_always_allowed`).

**Отрицание термина — не термин.** «Без флеш-дро», «нет флеш-дро», «не
флеш-дро» метки не требуют (`test_a_negated_term_is_not_the_term`).

**Граница, а не гарантия.** Допущение, термины и лучшая линия проверяются по
явным оборотам; что сказано другими словами, ловит судья (спека №2).

**Обозначения карт — не числа.** `A5s`, `T9o`, `KQ`, карта с мастью, ранги борда
вырезаются ДО сверки чисел. Чисто цифровая пара (`22`…`99`) от суммы шаблоном
не отличима (тех-спека §5.7); решение — она всегда число, даже если герой
держит эту пару (`test_a_bare_digit_pair_is_always_a_number`): выдуманная
сумма, совпавшая с парой героя, иначе прошла бы без проверки. Карту модель
пишет мастями или словами («пара двоек»); голая «22» вместо карты уходит на
переписывание — принятая цена. Цифры борда без разделителя («258») — сумма, а
не борд (`test_digits_without_separators_are_not_a_board`).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from harness.contracts import (
    AnalysisResult,
    DrawKind,
    EnrichedHand,
    PostflopLineDetail,
    Street,
    class_of,
    postflop_line_detail,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import error_words_in, unsupported_numbers
from harness.explanation.hand_replay import cards_text, hand_replay
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
_DIGITS_ONLY_RE = re.compile(r"\d+")
# Без разделителя ранги борда слипаются, только если рядом буква («KQ7»): три
# цифры подряд («258») — сумма, а не борд 2-5-8.
_BOARD_JOIN = r"(?:[-–]|(?<=[TJQKA])|(?=[TJQKA]))"
_BOARD_RE = re.compile(_EDGE_L + _RANK + r"(?:" + _BOARD_JOIN + _RANK + r"){2,4}" + _EDGE_R)
# Точка внутри числа («6.6») предложение не кончает.
_SENTENCE_RE = re.compile(r"(?:[^.!?\n]|\.(?=\d))+[.!?]?")
# Утверждение о сопернике (решение владельца 2026-10-13, контроллер, правка 1):
# предложение, которое (а) говорит о сопернике и (б) говорит о том, что он держит
# или сделает, несёт «(допущение)». Ложный отказ здесь дешевле ложного «прошло»:
# судьи в проде нет. До поиска (а) и (б) из предложения вырезается то, что
# держится на расчёте: действие, привязанное к доле или к «N из M» («сбрасывать
# 33.3% диапазона», «сбрасывало на с-бет 30 из 50»), и метаречь («про его
# диапазон это мало говорит»).
_W = r"(?<![а-яёa-z])"
_WE = r"(?![а-яё])"
_POSITION = r"(?<![A-Za-z\d.])(?<![\d.]\s)(?:BB|SB|BTN|CO|HJ|LJ|MP|UTG(?:\+\d)?)(?![A-Za-z])"
_OPPONENT_RE = re.compile(
    _W + r"(?:он|его|ему|него|нему|ним|нём)" + _WE
    + r"|соперник|оппонент|противник|регуляр"
    + r"|" + _W + r"пол(?:е|я|ю|ем)" + _WE
    + r"|так(?:ие|их|им|ой|ого)\s+игрок"
    + r"|" + _POSITION,
    re.IGNORECASE,
)
_ACTION = r"сброс|сбрас|сда[её]т|сдаст|сдаю|сдава|фолд|сфолд|защит|защищ|колл|заколл|рейз|блеф"
_HOLDING_RE = re.compile(
    _ACTION + r"|диапазон|рендж|держ|бродве|капн|кэпн|капирован"
    + r"|" + _W + r"пар(?:а|ы|у|ой|е|ам|ами|ах)?" + _WE
    # Рука словом-классом: `Jx`, `AK`, `A5s`, `98s` — с буквой; голые цифры («33»
    # из «33.3%») — не рука.
    + r"|(?<![\w.])(?-i:[2-9TJQKA]x|[TJQKA][2-9TJQKA][so]?|[2-9][TJQKA][so]?|[2-9]{2}[so])"
    r"(?![\w])",
    re.IGNORECASE,
)
# Число расчёта: доля или «N из M».
_CALC_NUMBER = r"\d+(?:[.,]\d+)?\s*%|\d+\s+из\s+\d+"
# Действие, за которым через пять слов или меньше стоит число расчёта, — порог
# или измерение, а не прогноз; вместе с хвостом «диапазона», «фолдов», «эквити».
_BACKED_RE = re.compile(
    r"(?:" + _ACTION + r")[^\s.,;:!?]*(?:[\s,]+[^\s.;:!?]+){0,5}?[\s,]+(?:"
    + _CALC_NUMBER
    + r")(?:\s+(?:его\s+)?(?:диапазон|рендж|фолд|эквити|случа|раз)[а-яё]*)?"
    + r"|(?:" + _CALC_NUMBER + r")\s+(?:его\s+)?(?:диапазон|рендж|фолд)[а-яё]*",
    re.IGNORECASE,
)
_META_RANGE_RE = re.compile(
    r"(?:про|о|об)\s+(?:его\s+|их\s+)?(?:диапазон|рендж)[а-яё]*(?:\s+[^\s.,;:!?]+){0,2}?"
    r"\s+(?:мало|ничего\s+не|почти\s+ничего\s+не)\s+говорит"
    r"|(?:мало|ничего\s+не|почти\s+ничего\s+не)\s+говорит\s+(?:про|о|об)\s+"
    r"(?:его\s+|их\s+)?(?:диапазон|рендж)[а-яё]*"
    r"|расч[её]т[а-яё]*\s+диапазон[а-яё]*",
    re.IGNORECASE,
)


def _opponent_claim(sentence: str) -> bool:
    """Предложение говорит, что соперник держит или сделает, не опираясь на
    расчёт (`test_a_claim_about_what_the_opponent_holds_or_does_needs_the_mark`)."""
    rest = _META_RANGE_RE.sub(" ", sentence)
    previous = None
    while previous != rest:
        previous, rest = rest, _BACKED_RE.sub(" ", rest)
    return bool(_OPPONENT_RE.search(rest) and _HOLDING_RE.search(rest))


# Лучшая линия на флопе и тёрне (решение владельца 2026-10-13). «Стоило» убрано:
# это слово вопроса игрока («стоило ставить?»), а не вердикт.
_BEST_LINE_RE = re.compile(
    r"лучш[а-яё]*\s+лини|лучше было|было лучше|лучше бы|правильн[а-яё]*\s+было"
    r"|правильн[а-яё]*\s+лини|следовало|надо было|нужно было"
    r"|(?<![A-Za-z])GTO(?![A-Za-z])|(?<![А-Яа-яЁё])ГТО(?![А-Яа-яЁё])",
    re.IGNORECASE,
)
# Вывод о прибыльности: голый — только по решённому порогу в разговоре; с долей и
# условием («окупается от 19.4% эквити», «при 24% фолдов») — арифметика (правка 4).
_PROFIT_RE = re.compile(_W + r"в\s+(?:плюсе|минусе)(?![а-яё])|окуп[а-яё]*", re.IGNORECASE)
_ARITHMETIC_RE = re.compile(
    _W + r"(?:если|при|от|когда|чтобы|чаще|реже)" + _WE + r"[^.!?]*?\d+(?:[.,]\d+)?\s*%",
    re.IGNORECASE,
)
# Отрицание перед термином: «без флеш-дро», «нет …», «не …», «ни …».
_NEGATION_BEFORE_RE = re.compile(_W + r"(?:без|нет|не|ни)\s+$", re.IGNORECASE)
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


_DRAW_WORD: dict[DrawKind, str] = {
    DrawKind.FLUSH: "флеш",
    DrawKind.OPEN_ENDED: "двусторонний стрит",
    DrawKind.DOUBLE_GUTSHOT: "двойной гатшот",
    DrawKind.GUTSHOT: "гатшот",
}
_BACKDOOR_WORD = {"flush": "флеш", "straight": "стрит"}


def _outs_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "аут"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "аута"
    return "аутов"


def _approx(share: float, book: NumberBook) -> str:
    """Константа владельца (бэкдоры, оверкарты) — с «≈», как в разборе."""
    return f"≈{book.pct(100.0 * share, trim=True)}%"


def _hero_equity_line(line: PostflopLineDetail, street: Street, book: NumberBook) -> str | None:
    """Дро, бэкдоры и оверкарты героя из посчитанного разбора — одной строкой, числа
    через книгу (`test_the_context_carries_the_hero_draws_backdoors_and_overcards`).
    Нет ни одного — строки нет."""
    parts: list[str] = []
    draw = line.draw
    if draw is not None:
        kinds = " + ".join(_DRAW_WORD[kind] for kind in draw.kinds)
        outs = len(draw.outs)
        chance = f"собрать на следующей улице {book.pct(100.0 * draw.hit_next, trim=True)}%"
        if draw.hit_by_river is not None:
            chance += f", к риверу {book.pct(100.0 * draw.hit_by_river, trim=True)}%"
        parts.append(f"дро — {kinds}, {book.count(outs)} {_outs_word(outs)}, {chance}")
    if line.backdoors:
        named = ", ".join(
            f"{_BACKDOOR_WORD[b.kind]} {_approx(b.approx, book)}" for b in line.backdoors
        )
        parts.append(f"бэкдор — {named}")
    over = line.overcards
    if over is not None:
        chance = (
            f"{_approx(over.approx_by_river, book)} к риверу"
            if over.approx_by_river is not None
            else f"{_approx(over.approx_next, book)} на следующей улице"
        )
        parts.append(f"оверкарты — {cards_text(over.cards)}, {chance}, если пара будет лучшей")
    if not parts:
        return None
    return f"Рука героя, {_STREET_NAME[street]}: {'; '.join(parts)}."


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
    метки классификатора всех точек и дро, бэкдоры, оверкарты героя из них.
    Разрешены числа реплея без обозначений
    карт: ранг карты героя или борда числом не становится
    (`test_a_card_rank_of_the_hand_is_not_an_allowed_number`). Идентификаторы в
    заголовке точки выжимки не разрешены, как в выжимке вердикта
    (`test_the_context_registers_every_number_it_prints`). Без судимых точек
    выжимки вердикта нет — только номер раздачи и это словами, без служебного
    «судимых точек 0» (`test_an_unjudged_hand_prints_no_service_count`)."""
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
    if result.ranked:
        verdict = verdict_digest(result)
        verdict_text, verdict_allowed = verdict.text, verdict.allowed
    else:
        verdict_text = (
            f"Раздача {book.token(result.hand_no)}.\nРешений с вердиктом в этой раздаче нет."
        )
        verdict_allowed = frozenset()
    labels: set[str] = set()
    label_lines: list[str] = []
    equity_lines: list[str] = []
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
        equity = _hero_equity_line(line, point.street, book)
        if equity is not None:
            equity_lines.append(equity)
    # Две точки одной улицы с одной рукой дали бы одну строку дважды.
    equity_text = "\n".join(dict.fromkeys(equity_lines))
    text = "\n\n".join(
        part
        for part in (replay, verdict_text, "\n".join(label_lines), equity_text)
        if part
    )
    return AgentContext(
        text=text,
        allowed=book.allowed | verdict_allowed,
        labels=frozenset(labels),
        hero_class=hero_class,
        board_ranks=board,
    )


def strip_card_tokens(
    text: str, *, hero_class: str | None = None, board_ranks: Sequence[str] = ()
) -> str:
    """Текст без обозначений карт — то, что идёт в сверку чисел.

    Вырезается только однозначное: карта с мастью, класс с буквой (`A5s`,
    `T9o`, `KQ`, `98s`), ранги борда через дефис. Две цифры без буквы и масти
    (`22`) остаются числом всегда. `hero_class` сохранён ради интерфейса
    (`AgentContext.hero_class`, `check_agent_answer`) и на вырезание не влияет.
    """
    text = _SUITED_CARD_RE.sub(" ", text)
    board = Counter(board_ranks)

    def board_sub(match: re.Match[str]) -> str:
        ranks = ["T" if rank == "10" else rank for rank in re.findall(_RANK, match.group(0))]
        has_letter = any(rank in "TJQKA" for rank in ranks)
        if (board and not (Counter(ranks) - board)) or (has_letter and not board):
            return " "
        return match.group(0)

    def class_sub(match: re.Match[str]) -> str:
        return match.group(0) if _DIGITS_ONLY_RE.fullmatch(match.group(0)) else " "

    return _CLASS_RE.sub(class_sub, _BOARD_RE.sub(board_sub, text))


def _says_term(pattern: re.Pattern[str], text: str) -> bool:
    """Термин сказан хотя бы раз без отрицания перед ним."""
    return any(
        not _NEGATION_BEFORE_RE.search(text[: match.start()]) for match in pattern.finditer(text)
    )


def check_agent_answer(
    answer: str,
    *,
    allowed: frozenset[float],
    labels: frozenset[str],
    leaks_called: bool,
    hero_class: str | None = None,
    board_ranks: Sequence[str] = (),
    threshold_decided: bool = False,
) -> list[str]:
    """Нарушения ответа словами; пустой список — ответ прошёл (спека §6).

    `threshold_decided` — в разговоре есть `frequency_vs_threshold` с выводом выше
    или ниже порога: только тогда «в плюсе» / «в минусе» допустимы выводом.
    """
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
        if _opponent_claim(sentence) and ASSUMPTION_MARK not in sentence:
            violations.append(
                f"утверждение о том, что соперник держит или сделает, без метки "
                f"{ASSUMPTION_MARK} — «{quoted}»"
            )
        if _BEST_LINE_RE.search(sentence) and _FLOP_TURN_RE.search(sentence):
            violations.append(f"лучшая линия на флопе или тёрне — «{quoted}»")
        if (
            _PROFIT_RE.search(sentence)
            and not _ARITHMETIC_RE.search(sentence)
            and not threshold_decided
        ):
            violations.append(
                "«в плюсе», «в минусе» или «окупается» выводом без расчёта "
                f"frequency_vs_threshold с выводом против порога — «{quoted}»"
            )
    for term, (pattern, needed) in TERM_LABELS.items():
        if _says_term(pattern, text) and not (needed & labels):
            violations.append(f"термин «{term}» без такой метки классификатора")
    if _LEAK_RE.search(text) and not leaks_called:
        violations.append("слово «лик» без расчёта leaks")
    errors = error_words_in(text)
    if errors:
        violations.append(f"решение названо ошибкой — {errors}")
    return violations
