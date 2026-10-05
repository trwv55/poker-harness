"""Спайк зрения: один скриншот -> модель -> сырой ответ, без обвязки.

Отвечает на один вопрос: виновата модель или мы. Ни бота, ни очереди, ни БД, ни
валидатора — только картинка, промпт и схема. Печатает три среза подряд:

1. **Сырой ответ инструмента** — что модель реально вернула, до валидации. Здесь
   видно то, чего не видно больше нигде: ответ, завёрнутый в контейнерный ключ
   (`params`, `$PARAMETER_NAME`), пустые поля, лишние ключи.
2. **Чтение после валидации** — во что схема превратила этот ответ. Расхождение
   между (1) и (2) и есть «мы потеряли прочитанное».
3. **Сборку и контрольные суммы** — `RawHand` и что сказала каждая проверка.

Задача 8 плана помечена «выкидной», спайк выкинули, и каждая диагностика зрения
начиналась с написания скрипта заново. Этот остаётся.

    uv run python scripts/vision_probe.py screen.png
    uv run python scripts/vision_probe.py screen.png --model anthropic:claude-opus-5
    uv run python scripts/vision_probe.py screen.png --nickname НИК_В_РУМЕ --json out.json
    uv run python scripts/vision_probe.py screen.png --repeat 3   # разброс между прогонами

`--repeat` печатает по строке на прогон: сколько игроков прочитано и какие
проверки провалились. Это замер воспроизводимости — температура вызова не
задана, и один скрин читается по-разному от прогона к прогону.

Промпт берётся с диска (`parsers/prompts/vision.md`) и в вывод НЕ печатается:
он закрыт политикой публикации. Ключ — из окружения или `.env`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _load_env() -> None:
    """`.env` в окружение, если ключа там ещё нет, — чтобы звать без `export`."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = _ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _tool_args(messages: Any) -> dict[str, Any] | None:
    """Аргументы вызова инструмента — тот самый сырой ответ модели."""
    for message in messages:
        for part in getattr(message, "parts", []):
            if type(part).__name__ == "ToolCallPart":
                args = part.args
                if isinstance(args, str):
                    try:
                        return json.loads(args)
                    except json.JSONDecodeError:
                        return {"<не разобрано как JSON>": args}
                if isinstance(args, dict):
                    return args
    return None


def _unwrapped(args: dict[str, Any]) -> dict[str, Any] | None:
    """Если ответ завёрнут в один чужой ключ — вернуть содержимое конверта."""
    from harness.contracts import VisionReading

    if len(args) != 1:
        return None
    (key, value), = args.items()
    if key in VisionReading.model_fields or not isinstance(value, dict):
        return None
    return value


async def _one_run(
    image: Path,
    model: str,
    temperature: float | None = None,
    seed: int | None = None,
    provider: str | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Один вызов модели: вернуть (чтение, сырые аргументы инструмента).

    `temperature=None` — как в проде: настройка не передаётся вовсе и действует
    умолчание провайдера. Ноль передаётся явно, чтобы мерить воспроизводимость.

    `seed` и `provider` нужны вместе с нулевой температурой и без них замер
    воспроизводимости ничего не значит. Температура 0 у многих хостеров не даёт
    полного детерминизма без явного seed, а на MoE-моделях разброс может идти от
    маршрутизации экспертов и батчинга НА СТОРОНЕ ХОСТЕРА — тогда один и тот же
    запрос, ушедший к разным провайдерам одной модели, вернёт разное. `provider`
    закрепляет хостера через маршрутизацию OpenRouter (`allow_fallbacks: false`):
    если закреплённый недоступен, честнее отказ, чем тихая подмена.
    """
    from pydantic_ai import Agent, BinaryContent
    from pydantic_ai.settings import ModelSettings

    from harness.contracts import VisionReading
    from harness.parsers.vision_adapter import read_prompt
    from harness.platform.llm import _sniff_image_media_type

    data = image.read_bytes()
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    if provider is not None:
        options["extra_body"] = {"provider": {"order": [provider], "allow_fallbacks": False}}
    settings = ModelSettings(**options) if options else None
    agent: Agent[None, VisionReading] = Agent(
        model, output_type=VisionReading, retries=0, model_settings=settings
    )
    result = await agent.run(
        [read_prompt(), BinaryContent(data=data, media_type=_sniff_image_media_type(data))]
    )
    return result.output, _tool_args(result.all_messages())


async def _probe(
    image: Path,
    model: str,
    nickname: str,
    dump: Path | None,
    temperature: float | None,
    seed: int | None,
    provider: str | None,
) -> int:
    from harness.parsers.vision_adapter import (
        match_hero,
        reading_to_raw,
        run_checks,
    )
    from harness.platform.llm import _sniff_image_media_type

    data = image.read_bytes()
    print(f"КАРТИНКА  {image}  {len(data)} байт  {_sniff_image_media_type(data)}")
    print(f"МОДЕЛЬ    {model}\n")

    reading, args = await _one_run(image, model, temperature, seed, provider)

    print("─" * 72)
    print("1. СЫРОЙ ОТВЕТ МОДЕЛИ (до валидации)")
    print("─" * 72)
    if args is None:
        print("  вызова инструмента нет — модель ответила текстом")
    else:
        print(f"  ключи верхнего уровня: {list(args)}")
        inner = _unwrapped(args)
        if inner is not None:
            print("  !! ОТВЕТ В КОНВЕРТЕ: всё содержимое лежит под одним чужим ключом")
            print(f"     внутри конверта: {list(inner)[:12]}")
        if dump is not None:
            dump.write_text(json.dumps(args, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  сохранён целиком: {dump}")

    print()
    print("─" * 72)
    print("2. ЧТЕНИЕ ПОСЛЕ ВАЛИДАЦИИ (то, что получил бы конвейер)")
    print("─" * 72)
    print(
        f"  игроков={len(reading.players)}  действий={len(reading.actions)}  "
        f"борд={reading.board}  рука={reading.hand_no!r}"
    )
    print(f"  не_рука={reading.not_a_hand}  отказ={reading.refusal_reason!r}")
    if not reading.players and not reading.not_a_hand:
        print("  !! ПУСТОЕ ЧТЕНИЕ: ни игроков, ни отказа")
        if args:
            print("     а в сыром ответе содержимое есть — значит потеряли мы, а не модель")
        return 1

    print()
    print("─" * 72)
    print("3. СБОРКА И КОНТРОЛЬНЫЕ СУММЫ")
    print("─" * 72)
    hero, hero_check = match_hero(nickname, [p.nickname for p in reading.players if p.nickname])
    raw, built = reading_to_raw(reading, hero_nickname=hero, source_ref=str(image))
    print(f"  герой={hero!r}  полнота={raw.completeness}  sb={raw.sb} bb={raw.bb} ante={raw.ante}")
    print(f"  мест={len(raw.seats)}  действий={len(raw.actions)}  борд={raw.boards}")
    print()
    for check in run_checks(reading, raw, hero_check, built):
        mark = " OK  " if check.passed else "ПРОВАЛ"
        print(f"  {mark} {check.name:10} {check.detail}")
        if not check.passed and check.options:
            print(f"         варианты для вопроса игроку: {check.options}")
    return 0


async def _repeat(
    image: Path,
    model: str,
    nickname: str,
    times: int,
    temperature: float | None,
    seed: int | None,
    provider: str | None,
) -> int:
    """Разброс между прогонами: температура не задана, чтение не воспроизводимо."""
    from harness.parsers.vision_adapter import match_hero, reading_to_raw, run_checks

    shown = "умолчание провайдера" if temperature is None else f"temperature={temperature}"
    if seed is not None:
        shown += f", seed={seed}"
    if provider is not None:
        shown += f", провайдер закреплён: {provider}"
    print(f"РАЗБРОС: {times} прогонов одной картинки через {model} ({shown})\n")
    for attempt in range(1, times + 1):
        reading, args = await _one_run(image, model, temperature, seed, provider)
        envelope = "конверт" if args and _unwrapped(args) is not None else "—"
        if not reading.players:
            print(f"  {attempt}. пустое чтение, обёртка: {envelope}")
            continue
        hero, hero_check = match_hero(
            nickname, [p.nickname for p in reading.players if p.nickname]
        )
        raw, built = reading_to_raw(reading, hero_nickname=hero, source_ref=str(image))
        failed = [c.name for c in run_checks(reading, raw, hero_check, built) if not c.passed]
        print(
            f"  {attempt}. игроков={len(reading.players)} обёртка: {envelope} "
            f"провалено: {failed or 'ничего'}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Спайк зрения: скрин -> модель -> сырой ответ")
    parser.add_argument("image", type=Path, help="путь к скриншоту")
    parser.add_argument(
        "--model",
        default=None,
        help="строка провайдера; по умолчанию LLM_VISION_MODEL из окружения",
    )
    parser.add_argument("--nickname", default="", help="ник героя в руме (для опознания героя)")
    parser.add_argument("--json", dest="dump", type=Path, help="куда сохранить сырой ответ")
    parser.add_argument("--repeat", type=int, default=0, help="замер разброса: N прогонов")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="передать температуру явно; без флага — как в проде, умолчание провайдера",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="зафиксировать seed (нужен вместе с температурой)"
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="закрепить хостера в OpenRouter без подмены, напр. DeepInfra",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"нет файла: {args.image}", file=sys.stderr)
        return 2

    _load_env()
    import os

    model = args.model or os.environ.get("LLM_VISION_MODEL")
    if not model:
        print("не задана модель: --model или LLM_VISION_MODEL", file=sys.stderr)
        return 2

    if args.repeat:
        return asyncio.run(
            _repeat(
                args.image,
                model,
                args.nickname,
                args.repeat,
                args.temperature,
                args.seed,
                args.provider,
            )
        )
    return asyncio.run(
        _probe(
            args.image,
            model,
            args.nickname,
            args.dump,
            args.temperature,
            args.seed,
            args.provider,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
