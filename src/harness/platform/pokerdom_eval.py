"""Local Pokerdom experiment. Private artifacts, one actual attempt per planned read."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from harness.contracts.pokerdom_vision import (
    PokerdomObservation,
    PokerdomReadingV2,
    decode_pokerdom_reading,
)
from harness.parsers.pokerdom_vision import PROMPT_PATH, process_pokerdom, read_pokerdom
from harness.platform.config import Config
from harness.platform.llm import LLM
from harness.presentation.pokerdom_vision import pokerdom_report

MODELS = ("openrouter:openai/gpt-6.1-terra", "openrouter:qwen/qwen4-max")
PROMPT = PROMPT_PATH


def _stable_configuration_error(exc: BaseException) -> bool:
    cause: BaseException | None = exc
    while cause is not None:
        if getattr(cause, "status_code", None) in {400, 401, 403, 404, 422}:
            return True
        cause = cause.__cause__
    return False


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def load_case(path: Path) -> tuple[dict, PokerdomObservation]:
    case = json.loads(path.read_text())
    return case, decode_pokerdom_reading(case["reading"])


def _private_output(path: Path) -> Path:
    resolved = path.resolve()
    root = Path.cwd().resolve()
    if not any(resolved.is_relative_to(root / prefix) for prefix in ("evals/vision", "reports")):
        raise ValueError("output must be inside ignored evals/vision/ or reports/")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


async def run(args: argparse.Namespace) -> int:
    out = _private_output(Path(args.out))
    models = tuple(args.model) if args.model else MODELS
    cases = [load_case(path) for path in sorted(Path(args.cases).glob("*.json"))
             if not any(part in path.stem for part in (".raw", ".enriched", ".reading"))]
    if args.case:
        selected = [args.case] if isinstance(args.case, str) else args.case
        cases = [(case, reading) for case, reading in cases if case["case_id"] in selected]
    if not cases:
        raise ValueError("no cases")
    if args.offline:
        passed = True
        for case, reading in cases:
            result = process_pokerdom(reading, nickname=args.nickname or case.get("hero_nickname"))
            passed = passed and result.replay_status == "passed"
            _write(out / f"{case['case_id']}.json", result.model_dump(mode="json"))
            (out / f"{case['case_id']}.txt").write_text(pokerdom_report(result))
        return 0 if passed else 1
    if any(out.iterdir()):
        raise ValueError("online run requires a new empty output directory; preserve prior attempts")
    # Preflight all input bytes and hashes before opening DB or making paid calls.
    inputs = []
    for case, reading in cases:
        source = case["source"]
        if not source.get("image_path") or not source.get("image_sha256"):
            raise ValueError(f"{case['case_id']}: original image path/hash not linked")
        image = Path(source["image_path"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != source["image_sha256"]:
            raise ValueError(f"{case['case_id']}: image hash mismatch")
        inputs.append((case, reading, image))
    if not args.terra_provider or not args.qwen_provider or args.repeats < 1:
        raise ValueError("fixed Terra/Qwen hosters and positive repeats required")
    from harness.platform.eval_runner import _call_costs, _open_llm
    cfg = Config.from_env()
    _, trace_id, sessions = await _open_llm(cfg, kind="pokerdom_vision")
    prompt = PROMPT.read_text()
    manifest = {"models": models, "repeats": args.repeats, "trace_id": trace_id,
                "schema_version": 2, "schema_sha256": hashlib.sha256(json.dumps(
                    PokerdomReadingV2.model_json_schema(), sort_keys=True).encode()).hexdigest(),
                "prompt_version": "pokerdom-v2", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "hosters": [args.terra_provider, args.qwen_provider],
                "provider_fallbacks": False, "automatic_retries": False, "cache": False,
                "sampling": "provider default", "reasoning": {"enabled": False},
                "tool_choice": "auto", "strict_tools": False, "sdk_retries": 0,
                "cases": [{"id": c["case_id"], "source": c["source"]} for c, _, _ in inputs]}
    source_paths = [Path(__file__), Path(__file__).with_name("llm.py"),
                    Path(__file__).parents[1] / "parsers/pokerdom_vision.py",
                    Path(__file__).parents[1] / "contracts/pokerdom_vision.py"]
    manifest["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source_paths}
    manifest["git_head"] = (await asyncio.to_thread(
        subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True)).strip()
    manifest["working_tree_snapshot"] = "source hashes identify local code; git_head alone may be older"
    _write(out / "manifest.json", manifest)
    summaries = []
    for model in models:
        provider = args.terra_provider if model == MODELS[0] else args.qwen_provider
        llm = LLM(cfg, sessions, model_override=model)
        configuration_failed = False
        for case, expected, image in inputs:
            if configuration_failed:
                break
            for repeat in range(1, args.repeats + 1):
                if configuration_failed:
                    break
                name = f"{case['case_id']}-{model.split('/')[-1]}-{repeat}"
                attempt: dict[str, Any] = {"case_id": case["case_id"], "model": model,
                    "planned_read": repeat, "provider_requested": provider, "schema_valid": False}
                # Persist before calling, including failures and interruptions.
                _write(out / f"{name}.attempt.json", attempt)
                started = time.monotonic()
                def observe(response: dict, record: dict = attempt, label: str = name) -> None:
                    record.update(response)
                    usage = response.get("response", {}).get("usage", {})
                    record["provider_reported_cost_usd"] = response.get("response", {}).get(
                        "provider_details", {}).get("cost")
                    record.update(tokens_in=usage.get("input_tokens"),
                                  tokens_out=usage.get("output_tokens"),
                                  cost_estimate_usd=usage.get("cost"))
                    _write(out / f"{label}.attempt.json", record)
                try:
                    reading, meta = await read_pokerdom(llm, image, trace_id=trace_id,
                                                        provider=provider, observer=observe)
                    attempt.update(schema_valid=True, model_returned=meta.model, read_ms=meta.latency_ms,
                        tokens_in=meta.tokens_in, tokens_out=meta.tokens_out, cost_estimate_usd=meta.cost_usd)
                    replay_started = time.monotonic()
                    result = process_pokerdom(reading,
                        nickname=args.nickname or case.get("hero_nickname"), expected=expected)
                    attempt["replay_ms"] = round((time.monotonic()-replay_started)*1000)
                    attempt.update(field_matches=result.field_matches,
                        missing_events=result.missing_events, added_events=result.added_events,
                        read_status=result.read_status, replay_status=result.replay_status,
                        truth_status=result.truth_status,
                        silent_error=result.replay_status == "passed" and result.truth_status == "mismatched")
                    _write(out / f"{name}.result.json", result.model_dump(mode="json"))
                    (out / f"{name}.txt").write_text(pokerdom_report(result))
                except Exception as exc:  # noqa: BLE001 — every failed attempt is a result
                    causes = []
                    cause = exc.__cause__
                    while cause is not None:
                        causes.append({"type": type(cause).__name__,
                            "status_code": getattr(cause, "status_code", None),
                            "body": getattr(cause, "body", None)})
                        cause = cause.__cause__
                    attempt["cause_chain"] = causes
                    configuration_failed = _stable_configuration_error(exc)
                    attempt["configuration_stop"] = configuration_failed
                    attempt["error"] = {"stage": "schema" if type(exc).__name__ == "LLMSchemaError"
                                        else "provider", "type": type(exc).__name__, "message": str(exc)}
                finally:
                    attempt["elapsed_ms"] = round((time.monotonic()-started)*1000)
                    _write(out / f"{name}.attempt.json", attempt)
                summaries.append(attempt)
                summary = summarize(summaries)
                summary.update(planned_reads=len(models)*len(inputs)*args.repeats,
                               not_attempted=len(models)*len(inputs)*args.repeats-len(summaries))
                _write(out / "summary.json", summary)
    _write(out / "costs.json", await _call_costs(sessions, trace_id))
    return 0


def _sum_known(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) if values else None


def summarize(attempts: list[dict]) -> dict:
    groups = {}
    for model in MODELS:
        for case_id in sorted({a["case_id"] for a in attempts}):
            rows = [a for a in attempts if a["model"] == model and a["case_id"] == case_id]
            counts = Counter(a.get("truth_status", "unscored") for a in rows)
            field_names = sorted({key for row in rows for key in row.get("field_matches", {})})
            groups[f"{model}/{case_id}"] = {"denominator": len(rows),
                "field_matches": {key: {"numerator": sum(row.get("field_matches", {}).get(key, False)
                                                       for row in rows), "denominator": len(rows)}
                                  for key in field_names},
                "missing_events": sum(len(row.get("missing_events", [])) for row in rows),
                "added_events": sum(len(row.get("added_events", [])) for row in rows),
                "read_status": dict(Counter(row.get("read_status", "error") for row in rows)),
                "schema_valid": sum(a["schema_valid"] for a in rows),
                "replay_passed": sum(a.get("replay_status") == "passed" for a in rows),
                "silent_errors": sum(bool(a.get("silent_error")) for a in rows),
                "truth": dict(counts), "elapsed_ms": [a["elapsed_ms"] for a in rows],
                "cost_estimate_usd": _sum_known(rows, "cost_estimate_usd"),
                "provider_reported_cost_usd": _sum_known(rows, "provider_reported_cost_usd"),
                "provider_cost_unknown": sum(a.get("provider_reported_cost_usd") is None for a in rows),
                "read_ms": [a.get("read_ms") for a in rows],
                "replay_ms": [a.get("replay_ms") for a in rows],
                "cost_unknown": sum(a.get("cost_estimate_usd") is None for a in rows)}
    return {"groups": groups, "attempts": len(attempts)}


def add_parser(sub: Any) -> None:
    parser = sub.add_parser("pokerdom-vision", help="local Pokerdom reading/replay experiment")
    parser.add_argument("--cases", default="evals/vision/pokerdom/runner-cases")
    parser.add_argument("--out", required=True)
    parser.add_argument("--case", action="append", help="selected case ID; repeat for several cases")
    parser.add_argument("--model", action="append", choices=MODELS, help="diagnostic subset; recorded explicitly")
    parser.add_argument("--nickname", default="")
    parser.add_argument("--offline", action="store_true", help="replay observations without model calls")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--terra-provider")
    parser.add_argument("--qwen-provider")
