"""Реплей канонической руки на движке PokerKit — механика без политики.

Модуль проигрывает руку по правилам NLHE и считает деньги сам: банк по улицам,
сайд-поты, стеки на конец руки. Записанным в источнике итогам (`Total Pot`,
`collected`) он не верит и на них не опирается — расхождение между своим
подсчётом и источником обязано остаться видимым, потому что именно оно является
сигналом о битых данных.

Физически невозможное (ход не того игрока, ставка, которую нельзя сделать,
несходящаяся сумма) записывается в `illegal_actions`, реплей на этом
прерывается. Судить руку в целом — не дело этого модуля: вердикт выносит
`harness.engine.validation`.
"""

from __future__ import annotations

from collections import deque

from pokerkit import Automation, Deck, NoLimitTexasHoldem
from pokerkit.state import State

from harness.contracts import (
    ActionKind,
    CanonicalAction,
    CanonicalHand,
    DecisionPoint,
    EngineReport,
    SidePot,
    Street,
)
from harness.normalizer import POSITIONS_BY_COUNT

# Улицы NLHE по индексу улицы в PokerKit (`State.street_index`).
_STREET_BY_INDEX: tuple[Street, ...] = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)

# Карты стандартной колоды в фиксированном порядке — детерминированный остаток
# для неизвестных карт оппонентов (на арифметику денег они не влияют).
_DECK: tuple[str, ...] = tuple(repr(card) for card in Deck.STANDARD)
_DECK_SET = frozenset(_DECK)

_HOLE_CARD_COUNT = 2

# Предел шагов реплея: страховка от зацикливания на битых данных.
_MAX_STEPS = 500

# Автоматизируем только механику без выбора: анте/блайнды, вскрытие/сброс на
# шоудауне, убийство проигравших рук и (в турнире и так фиктивный) выбор числа
# ранаутов. Остальное ведём сами:
#   * карты — из источника, а сжигаемые из своего остатка колоды, иначе PokerKit
#     тянет их из перетасованной колоды и может «сжечь» карту, которую источник
#     кладёт на стол;
#   * сбор ставок — чтобы увидеть ставки круга до того, как они уедут в банк
#     (см. `_collect_bets`);
#   * раздачу выигрышей — банк надо снять ДО неё.
_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.RUNOUT_COUNT_SELECTION,
)


class _CardSupply:
    """Раздатчик карт: занимает известные из источника, остальное берёт из колоды.

    Карту, которой в стандартной колоде нет или которая уже занята (битые данные,
    ошибка распознавания), подменяет свободной. Подмена не «чинит» руку — она лишь
    даёт движку валидное состояние; о самом дубле рассудит валидатор, читая
    исходную руку, а не этот раздатчик.
    """

    def __init__(self) -> None:
        self._used: set[str] = set()
        self._pool = deque(_DECK)

    def claim(self, cards: list[str]) -> list[str]:
        """Занять перечисленные карты, подменив недоступные."""
        return [self._claim_one(card) for card in cards]

    def draw(self) -> str:
        """Взять следующую свободную карту колоды."""
        while self._pool:
            card = self._pool.popleft()
            if card not in self._used:
                self._used.add(card)
                return card
        raise ValueError("колода исчерпана: раздать карту нечем")

    def _claim_one(self, card: str) -> str:
        if card in _DECK_SET and card not in self._used:
            self._used.add(card)
            return card
        return self.draw()


def _engine_seating(hand: CanonicalHand) -> list[str]:
    """Порядок игроков для PokerKit: индекс 0 — получатель первого блайнда.

    PokerKit назначает `raw_blinds_or_straddles` по индексам игроков, поэтому
    порядок мест надо задать от малого блайнда. Для хедз-апа порядок обратный:
    `State.get_effective_blind_or_straddle` в игре на двоих меняет блайнды
    местами, так что нулевой индекс там — большой блайнд, а кнопка (она же SB) —
    первый. Это расходится с порядком позиций нормалайзера, где HU начинается с
    кнопки.
    """
    rank = {position: i for i, position in enumerate(POSITIONS_BY_COUNT[len(hand.players)])}
    seating = sorted(hand.players, key=lambda player: rank[player.position])
    labels = [player.label for player in seating]
    if len(labels) == _HOLE_CARD_COUNT:
        labels.reverse()
    return labels


def _position(hand: CanonicalHand, label: str) -> str:
    return next(p.position for p in hand.players if p.label == label)


def _known_hole_cards(hand: CanonicalHand, label: str) -> list[str]:
    """Карты игрока, известные из источника: раздача плюс вскрытие."""
    cards = list(hand.dealt.get(label, []))
    for entry in hand.showdowns:
        if entry.label == label:
            cards += [card for card in entry.cards if card not in cards]
    return cards[:_HOLE_CARD_COUNT]


def _deal_cards(
    hand: CanonicalHand, seating: list[str], supply: _CardSupply
) -> tuple[dict[str, list[str]], dict[Street, list[str]]]:
    """Разложить известные карты и добрать неизвестные из остатка колоды."""
    boards = {street: supply.claim(cards) for street, cards in hand.boards.items()}
    known = {label: supply.claim(_known_hole_cards(hand, label)) for label in seating}
    holes = {
        label: cards + [supply.draw() for _ in range(_HOLE_CARD_COUNT - len(cards))]
        for label, cards in known.items()
    }
    return holes, boards


def _create_state(hand: CanonicalHand, seating: list[str]) -> State:
    stacks = {p.label: p.stack for p in hand.players}
    return NoLimitTexasHoldem.create_state(
        _AUTOMATIONS,
        # ante_trimming_status=False — анте в GG платит каждый игрок отдельно,
        # это не «анте баттона», которое подрезается под общий уровень.
        False,
        hand.ante,
        (hand.sb, hand.bb),
        hand.bb,
        [stacks[label] for label in seating],
        len(seating),
    )


def _playable(state: State, index: int) -> int:
    """Фишки игрока, которыми ещё играется текущий круг торговли.

    Остаток стека плюс уже поставленное в этом круге. Именно остаток стека сам по
    себе для глубины не годится: у игрока в олл-ине он равен нулю, и любой счёт
    «по остаткам» такого игрока просто не видит.

    Префлоп величина равна стартовому стеку за вычетом анте, и это следует из
    механики PokerKit, а не из данных: анте постится в `bets` и сметается
    `collect_bets()` ДО постановки блайндов, поэтому весь префлоп-круг `bets`
    несёт только блайнд и добровольное, а анте уже в банке. Равенство держится
    при любом развитии торговли, в том числе при `ante = 0` и у игрока, которому
    на анте не хватило стека (`get_effective_ante` — то же `min(анте, стек)`,
    что и в `classifier.py`).

    Та же величина зовётся в анализе `SeatSnapshot.stack_after_ante` — глубина,
    которой индексируется равновесие пуш-фолда. Одна точка на настоящей руке с
    анте 750 затянута тестом
    `tests/test_preflop_analysis.py::test_eff_stack_is_the_depth_the_model_indexes_by`;
    сверка на 2536 точках игрок×точка обеих фикстур (ноль расхождений) была
    разовым замером при этой правке, а не постоянной проверкой — держит здесь
    механика выше, а не тот замер.

    Постфлоп `bets` обнуляются на границе улицы, поэтому там это стек, с которым
    игрок вошёл в улицу: деньги прошлых улиц уже в банке и в глубину не входят.
    """
    return state.stacks[index] + state.bets[index]


def _effective_stack(state: State, actor: int, aggressor: int | None) -> int:
    """Глубина, на которую играется решение: против того, чью ставку отвечают.

    Раньше здесь стоял максимум остатков стека по всем живым оппонентам, и это
    был структурный дефект: у игрока в олл-ине остаток нулевой, поэтому максимум
    **никогда** его не выбирал — он всегда предпочитал любого, у кого фишки ещё
    есть. Герой, отвечающий на шов, мерился против игрока, которого в этом
    решении нет вовсе: шов на 2.1bb получал глубину 30.7bb по стеку постороннего
    и вылетал из пуш-фолд-зоны как «слишком глубокий».

    Оппонент — последний, кто ставил или повышал в текущем круге: именно его
    ставку решающий коллирует или сбрасывает, и именно его диапазоном спот
    моделируется (`analysis.classifier.TableState.aggressor` — тот же игрок,
    восстановленный из строк руки независимо от движка).

    **Какую из глубин модели повторяет этот гейт.** Модель считает две разные, и
    совпадение достигнуто с одной. Совпадает с `equilibrium_depth`
    (`analysis.preflop._facing_shove_verdict`: `min(герой, шовер)` по
    `stack_after_ante` — буквально это выражение) и с глубинами колл-моделей
    живых позади. НЕ совпадает с `shover_depth_bb` там же: тот индексирует
    пуш-диапазон шовера как `min(шовер, максимум по ВСЕМ местам, включая
    спасовавших)` — величина про то, насколько глубоко играл шовер, когда
    принимал своё решение, а не про то, на какую глубину играется решение героя.
    Асимметрия намеренная: `shover_depth_bb` — вход в модель чужого диапазона,
    трогать его этой правкой было прямо запрещено.

    Три случая, которые эта величина обязана держать:

    1. **Мультивей.** Герой отвечает на шов A, а живой B с глубоким стеком ещё в
       руке. Меряем против A. Цена ЭТОГО решения ограничена шовом A, и модель
       судит его против диапазона A. Проиграть герой может и B — но это не
       вопрос глубины, а вопрос допущения, и он уже задан
       отдельной осью: `analysis.preflop._facing_shove_verdict` пересчитывает EV
       со входом всех живых, кроме героя и шовера, и при сдвиге вердикта роняет
       зону в `assuming`. Учесть B ещё и в глубине значило бы применить одну
       поправку дважды и молча выбросить чистый пуш-фолд-спот вместо того, чтобы
       честно пометить его допущением.
    2. **Агрессора нет вовсе** — неоткрытый банк (до решающего только посты и
       лимпы) или круг, в котором ещё никто не ставил. Отвечать не на что,
       конкретного оппонента у решения нет, и величина возвращается к своему
       общему смыслу — потолок руки: больше, чем есть у самого глубокого живого
       оппонента, выиграть или проиграть нельзя. Одним числом такой спот и не
       описывается: против каждого живого глубина своя, и модель шова считает их
       по отдельности (`_unopened_verdict`: `min(глубина героя, глубина
       позади_i)` для каждого игрока позади). Одно число здесь — гейт
       применимости, и берётся самый глубокий: шов героя на 40bb остаётся
       решением на 40bb, даже если один из оппонентов позади сидит с 3bb. Взять
       самого короткого значило бы объявить пуш-фолдом любой спот, где за столом
       нашёлся хоть один короткий стек.
    3. **Постфлоп.** Числитель SPR — стек на входе в улицу (см. `_playable`), то
       есть штатный числитель SPR из учебника, а не «остаток после собственной
       ставки». Знаменателем как был, так и остаётся банк на момент решения:
       ставки текущей улицы в него входят, поэтому против ставки отношение выйдет
       ниже учебного. Это свойство `pot_before`, а не глубины, и разбирать его
       место — ступень 2 (постфлоп пока не оценивается вовсе).
    """
    if aggressor is not None and aggressor != actor and state.statuses[aggressor]:
        return min(_playable(state, actor), _playable(state, aggressor))
    rivals = [
        _playable(state, i) for i in state.player_indices if i != actor and state.statuses[i]
    ]
    return min(_playable(state, actor), max(rivals)) if rivals else _playable(state, actor)


def _decision_point(
    state: State,
    hand: CanonicalHand,
    seating: list[str],
    action: CanonicalAction,
    index: int,
    aggressor: int | None,
) -> DecisionPoint:
    """Снимок состояния перед решением игрока.

    `pot_before` — банк, за который решающий реально играет: вклад каждого игрока
    учитывается не полностью, а до потолка «сколько решающий вообще способен
    поставить» (уже поставленное плюс стек). Ставку сверх этого потолка выиграть
    нельзя — она вернётся поставившему, и складывать её в банк значило бы
    завысить шансы банка. На тестовой руке это даёт 14532, а `pot_before + to_call`
    — 14673, ровно тот мейн-пот, который записал рум.

    `eff_stack` — глубина решения против того, чью ставку отвечают, см.
    `_effective_stack`. `aggressor` — место последнего, кто ставил или повышал в
    этом круге, или `None`, если таких не было.

    `live_total` и `live_behind` — вход правила зоны доверия: сколько игроков ещё
    в руке и сколько из них ходят после решающего в текущем круге.
    """
    actor = seating.index(action.label)
    street_index = state.street_index
    assert street_index is not None
    street = _STREET_BY_INDEX[street_index]
    # -payoffs[i] — всё, что игрок вложил в руку к этому моменту: собранное в
    # банк плюс невыровненные ставки круга (анте и блайнды тоже).
    committed = [-state.payoffs[i] for i in state.player_indices]
    ceiling = committed[actor] + state.stacks[actor]
    pot_before = sum(min(amount, ceiling) for amount in committed)
    eff_stack = _effective_stack(state, actor, aggressor)
    spr = eff_stack / pot_before if street is not Street.PREFLOP and pot_before else None
    return DecisionPoint(
        index=index,
        street=street,
        label=action.label,
        position=_position(hand, action.label),
        to_call=state.checking_or_calling_amount or 0,
        pot_before=pot_before,
        eff_stack=eff_stack,
        eff_stack_bb=eff_stack / hand.bb,
        spr=spr,
        action=action,
        live_total=sum(state.statuses),
        live_behind=max(len(state.actor_indices) - 1, 0),
    )


def _apply(state: State, action: CanonicalAction) -> None:
    """Сыграть действие источника. Невозможное бросает `ValueError` из PokerKit."""
    match action.kind:
        case ActionKind.FOLD:
            state.fold()
        case ActionKind.CHECK | ActionKind.CALL:
            state.check_or_call()
        case ActionKind.BET | ActionKind.RAISE:
            state.complete_bet_or_raise_to(action.committed_after)


class _Replay:
    """Состояние одного прогона: собирается только внутри `replay`."""

    def __init__(self, hand: CanonicalHand) -> None:
        self.hand = hand
        self.seating = _engine_seating(hand)
        self.supply = _CardSupply()
        self.holes, self.boards = _deal_cards(hand, self.seating, self.supply)
        self.state = _create_state(hand, self.seating)
        self.pending = deque(hand.actions)
        self.illegal: list[str] = []
        self.points: list[DecisionPoint] = []
        self.pot_by_street: dict[Street, int] = {}
        self.side_pots: list[SidePot] = []
        self.final_pot: int | None = None
        self.forfeits: list[str] = []
        self._street_index = self.state.street_index
        self._uncalled = 0
        # Место последнего, кто ставил или повышал в текущем круге, — оппонент,
        # против которого меряется глубина решения (см. `_effective_stack`).
        # Ведём его реплеем, а не выводим из `state.bets`: там ставку агрессора
        # уравнивает коллер, и по одним суммам последний агрессор от уравнявшего
        # неотличим.
        self._aggressor: int | None = None

    def run(self) -> None:
        for _ in range(_MAX_STEPS):
            if not self._step():
                return
            self._note_street_boundary()
            if not self.state.status:
                self._note_unplayed()
                return
        self.illegal.append(f"реплей не завершился за {_MAX_STEPS} шагов")

    def _step(self) -> bool:
        """Один шаг реплея. `False` — дальше идти нельзя, причина уже записана."""
        state = self.state
        if self._is_forfeit(self.pending[0] if self.pending else None):
            self._forfeit(self.pending.popleft().label)
        elif state.can_collect_bets():
            self._collect_bets()
        elif state.can_burn_card():
            state.burn_card(self.supply.draw())
        elif state.can_deal_hole():
            self._deal_hole()
        elif state.can_deal_board():
            self._deal_board()
        elif state.actor_index is not None:
            return self._act()
        elif state.can_push_chips():
            self._collect_pot()
        elif state.can_pull_chips():
            state.pull_chips()
        else:
            self.illegal.append("реплей встал: состояние не предлагает следующей операции")
            return False
        return True

    def _is_forfeit(self, action: CanonicalAction | None) -> bool:
        """Пас от игрока, у которого не осталось фишек за спиной.

        Так GG записывает игрока, ушедшего в олл-ин **вынужденной** ставкой
        (обычно он сидит аут или отвалился по связи): рум считает его вышедшим из
        руки, а его фишки — мёртвыми деньгами в банке. Правила NLHE так не умеют —
        олл-ин игрок остаётся в руке и претендует на мейн-пот, — и PokerKit хода
        у него не спросит. Провенанс `hand_history` — это факт: отказ проиграть
        записанное румом действие дал бы ложный `reject` на настоящей руке
        игрока.

        Условие держим предельно узким, потому что цена ошибки — молча уехавший
        не туда банк. Мало того, что это явный `folds` при нулевом остатке
        стека: игрок должен был обнулиться **на анте с блайндом**, а не
        добровольной ставкой. Иначе лишняя строка `folds` после чьего-нибудь
        олл-ин-рейза (ровно то, что даёт сбой распознавания или баг парсера)
        вывела бы из руки живого игрока и отдала бы его банк сопернику.
        """
        if action is None or action.kind != ActionKind.FOLD:
            return False
        if action.label not in self.seating:
            return False
        index = self.seating.index(action.label)
        if not self.state.statuses[index] or self.state.stacks[index]:
            return False
        if not self._all_in_from_forced_bet(action.label):
            return False
        # Последнего оставшегося не выводим: пас всех до одного — не форфейт,
        # а битые данные, и разбираться с ними должен обычный путь.
        return sum(self.state.statuses) > 1

    def _all_in_from_forced_bet(self, label: str) -> bool:
        """Игроку хватило стека ровно на анте с блайндом — добровольно он не ставил."""
        index = self.seating.index(label)
        posted = self.state.get_effective_ante(index) + self.state.get_effective_blind_or_straddle(
            index
        )
        return self.state.starting_stacks[index] <= posted

    def _forfeit(self, label: str) -> None:
        """Исполнить записанный румом пас: игрок выходит из руки, вклад остаётся.

        `State.fold` для этого не годится: она требует, чтобы игрок был на ходу,
        и прямо утверждает `assert self.stacks[player_index]` — по правилам
        человек без фишек пасовать не может. Снимаем ровно тот флаг, который
        снимает сама PokerKit при пасе (`statuses`): игрок перестаёт быть
        участником банка (`State.pots` раздаёт поты только живым), а всё, что он
        успел вложить, остаётся в банке мёртвыми деньгами — как и у рума.

        Пробовал сделать это сбросом карт на шоудауне (`show_or_muck_hole_cards`
        со статусом `False`) — не работает: при олл-ин-ранауте PokerKit вскрывает
        карты ДО добора борда, и сброс оставляет за столом одного живого игрока,
        после чего `_begin_betting` следующей улицы падает на
        `assert len(effective_stacks) > 1` внутри `get_effective_stack`.
        """
        self.forfeits.append(label)
        self.state.statuses[self.seating.index(label)] = False

    def _note_unplayed(self) -> None:
        """Рука кончилась, а действия источника остались — это расхождение, не мелочь.

        Иначе лишние строки просто пропали бы, и рука разошлась бы с источником
        молча: банк сошёлся, а фишки уехали не тому.
        """
        for action in self.pending:
            self.illegal.append(
                f"действие источника осталось несыгранным: {action.raw_line}"
                f"{self._blind_all_in_hint(action.label)}"
            )

    def _blind_all_in_hint(self, label: str) -> str:
        """Подсказка про игрока, оставшегося без фишек ещё на блайндах.

        GG пишет такому игроку `folds`, хотя по правилам он уже в олл-ине и
        ходить не может, а его блайнд остаётся живым в банке. Это расхождение
        рума с правилами, а не битый парсер — диагностике стоит сказать прямо.
        """
        if label not in self.seating or not self._all_in_from_forced_bet(label):
            return ""
        return " (игрок ушёл в олл-ин ещё на анте с блайндом и ходить не может)"

    def _pot_now(self) -> int:
        """Банк без непоколленной ставки — как его считает рум.

        Когда все спасовали, PokerKit не забирает ставку последнего оставшегося
        в банк: она так и висит перед ним и вернётся ему при раздаче. Рум же
        пишет банк уже за вычетом этой ставки («Uncalled bet returned to»), и
        сверять надо именно с такой суммой.
        """
        return self.state.total_pot_amount - self._uncalled

    def _collect_bets(self) -> None:
        """Забрать ставки круга в банк, разобравшись с непоколленной ставкой.

        Когда все спасовали, PokerKit оставляет ставку последнего оставшегося
        висеть перед ним целиком — и непоколленную сдачу, и ту часть, которую
        оппоненты успели уравнять. Рум делит её надвое: сдачу возвращает
        («Uncalled bet returned to»), уравненную часть считает банком. Считаем
        так же: сдача — это превышение над вторым по величине вкладом круга.
        """
        state = self.state
        bets = list(state.bets)
        matched = 0
        survivor = 0
        if state.street is not None and sum(state.statuses) == 1:
            survivor = state.statuses.index(True)
            uncalled = max(bets[survivor] - sorted(bets)[-2], 0)
            self._uncalled += uncalled
            matched = bets[survivor] - uncalled
        state.collect_bets()
        # Состав банка снимаем сразу после сбора: к раздаче выигрышей PokerKit
        # схлопнет мейн и сайды в один пот победителя, и деление пропадёт.
        self.side_pots = [
            SidePot(amount=pot.amount, eligible=[self.seating[i] for i in pot.player_indices])
            for pot in state.pots
        ]
        if matched:
            self._add_matched(matched, self.seating[survivor])

    def _add_matched(self, matched: int, label: str) -> None:
        """Дописать в банк уравненную часть ставки последнего оставшегося."""
        if self.side_pots and self.side_pots[-1].eligible == [label]:
            self.side_pots[-1].amount += matched
        else:
            self.side_pots.append(SidePot(amount=matched, eligible=[label]))

    def _note_street_boundary(self) -> None:
        """Зафиксировать банк на границе улицы — накопительно, после сбора ставок."""
        if self.state.street_index == self._street_index:
            return
        if self._street_index is not None:
            street = _STREET_BY_INDEX[self._street_index]
            self.pot_by_street[street] = self._pot_now()
        self._street_index = self.state.street_index
        # Новая улица — торговля начинается заново, прежний агрессор к ней
        # отношения не имеет.
        self._aggressor = None

    def _deal_hole(self) -> None:
        index = self.state.hole_dealee_index
        assert index is not None
        self.state.deal_hole("".join(self.holes[self.seating[index]]), index)

    def _deal_board(self) -> None:
        street_index = self.state.street_index
        assert street_index is not None
        street = _STREET_BY_INDEX[street_index]
        needed = self.state.board_dealing_count or 0
        cards = self.boards.get(street, [])
        if len(cards) != needed:
            self.illegal.append(
                f"карты борда на улице {street}: ожидалось {needed}, в источнике {len(cards)}"
            )
            cards = [self.supply.draw() for _ in range(needed)]
        self.state.deal_board("".join(cards))

    def _act(self) -> bool:
        """Сыграть следующее действие источника. `False` — реплей прерван."""
        actor = self.state.actor_index
        assert actor is not None
        expected = self.seating[actor]
        if not self.pending:
            self.illegal.append(f"действия источника кончились, а ход за {expected}")
            return False

        action = self.pending.popleft()
        if action.label != expected:
            self.illegal.append(
                f"ход за {expected}, а источник ходит за {action.label}: "
                f"{action.raw_line}{self._blind_all_in_hint(action.label)}"
            )
            return False
        to_call = self.state.checking_or_calling_amount
        if action.kind == ActionKind.CHECK and to_call:
            self.illegal.append(f"чек при доплате {to_call}: {action.raw_line}")
            return False
        if action.label == self.hand.hero_label:
            self.points.append(
                _decision_point(
                    self.state, self.hand, self.seating, action, len(self.points), self._aggressor
                )
            )

        try:
            _apply(self.state, action)
        except ValueError as exc:
            self.illegal.append(f"{action.raw_line}: {exc}")
            return False
        # Ниже снятия точки решения (см. `self.points.append` выше), иначе
        # собственное повышение решающего попало бы в его же снимок и глубина
        # мерилась бы против него самого. Относительно `_apply` порядок роли не
        # играет.
        if action.kind in (ActionKind.BET, ActionKind.RAISE):
            self._aggressor = actor

        if action.kind != ActionKind.FOLD and self.state.bets[actor] != action.committed_after:
            self.illegal.append(
                f"сумма разошлась у {action.label}: движок {self.state.bets[actor]}, "
                f"источник {action.committed_after} ({action.raw_line})"
            )
            return False
        return True

    def _collect_pot(self) -> None:
        """Снять банк ДО раздачи выигрышей, затем раздать."""
        self.final_pot = self._pot_now()
        self.state.push_chips()

    def report(self) -> EngineReport:
        state = self.state
        return EngineReport(
            pot_by_street=self.pot_by_street,
            final_pot=self._pot_now() if self.final_pot is None else self.final_pot,
            side_pots=self.side_pots,
            # Прерванный реплей отдаёт неполные стеки: фишки остаются в банке.
            # Это честная частичная картина — «досчитывать» её движок не станет.
            stacks_end={
                label: state.stacks[i] + state.bets[i] for i, label in enumerate(self.seating)
            },
            decision_points=self.points,
            illegal_actions=self.illegal,
            forfeits=self.forfeits,
        )


def replay(hand: CanonicalHand) -> EngineReport:
    """Проиграть руку по правилам NLHE и вернуть посчитанные движком цифры."""
    run = _Replay(hand)
    run.run()
    return run.report()
