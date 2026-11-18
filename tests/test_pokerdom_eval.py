"""Experiment transport checks use synthetic model responses, never a model eval."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openrouter import OpenRouterModel

from harness.parsers.pokerdom_vision import read_pokerdom
from harness.platform.config import Config
from harness.platform.llm import LLM, LLMProviderError, LLMSchemaError, _build_model
from harness.platform.pokerdom_eval import _stable_configuration_error, summarize


class Output(BaseModel):
    value: str


class Limiter:
    @asynccontextmanager
    async def slot(self, **kwargs):
        yield


def facade(model):
    cfg = Config(llm_vision_model="", llm_vision_fallback_model="",
                 llm_max_concurrency=1, llm_max_per_minute=1000, llm_verdict_model="test:synthetic", database_url="unused", telegram_token="unused")
    with patch("harness.platform.llm.PgLimiter", return_value=Limiter()):
        llm = LLM(cfg, MagicMock(), model_override=FunctionModel(model))
    llm._log_started = AsyncMock(return_value=1)
    llm._log_finished = AsyncMock()
    llm._log_status = AsyncMock()
    return llm


async def test_experiment_logs_raw_schema_failure_without_retry():
    requests = []
    def model(messages, info):
        requests.append(messages)
        return ModelResponse(parts=[TextPart("invalid schema")],
                             provider_details={"downstream_provider": "synthetic-host"})
    llm = facade(model)
    observed = []
    with pytest.raises(LLMSchemaError):
        await llm("vision_extract", Output, prompt="synthetic", trace_id=1,
                  experiment_provider="synthetic-host", experiment_observer=observed.append)
    assert len(requests) == 1
    assert observed[0]["served_by"] == "synthetic-host"
    assert observed[0]["response"]["parts"][0]["content"] == "invalid schema"
    assert isinstance(llm._log_status, AsyncMock)
    llm._log_status.assert_awaited_once_with(1, "schema_error")


async def test_experiment_transport_failure_is_single_attempt():
    requests = []
    def model(messages, info):
        requests.append(messages)
        raise ModelHTTPError(status_code=503, model_name="synthetic")
    llm = facade(model)
    with pytest.raises(LLMProviderError):
        await llm("vision_extract", Output, prompt="synthetic", trace_id=1,
                  experiment_provider="synthetic-host")
    assert len(requests) == 1


async def test_experiment_success_keeps_response_and_host():
    def model(messages, info):
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {"value": "literal"})],
                             provider_details={"downstream_provider": "synthetic-host"})
    llm = facade(model)
    observed = []
    result, _ = await llm("vision_extract", Output, prompt="synthetic", trace_id=1,
        experiment_provider="synthetic-host", experiment_observer=observed.append)
    assert result.value == "literal"
    assert len(observed) == 1 and observed[0]["call_id"] == 1


def test_fixed_host_route_blocks_fallbacks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    model = _build_model("openrouter:qwen/qwen4-max", fixed_provider="synthetic-host")
    assert isinstance(model, OpenRouterModel) and model.settings is not None
    route = model.settings.get("openrouter_provider")
    assert route is not None
    assert model.client.max_retries == 0
    assert model.profile.get("openai_supports_tool_choice_required") is False
    assert model.profile.get("openai_supports_strict_tool_definition") is False
    assert model.settings.get("openrouter_reasoning") == {"enabled": False}
    assert route["only"] == ["synthetic-host"] and route["allow_fallbacks"] is False


async def test_reader_receives_no_hero_or_expected_and_no_history(monkeypatch, tmp_path):
    path = tmp_path / "prompt.md"
    path.write_text("synthetic format instruction")
    monkeypatch.setattr("harness.parsers.pokerdom_vision.PROMPT_PATH", path)
    llm = AsyncMock(return_value=(None, None))
    image = b"synthetic image bytes"
    await read_pokerdom(llm, image, trace_id=2, provider="host", observer=None)
    kwargs = llm.call_args.kwargs
    assert set(kwargs) == {"prompt", "images", "trace_id", "experiment_provider", "experiment_observer"}
    assert kwargs["images"] == [image]
    assert kwargs["prompt"] == "synthetic format instruction"


def test_summary_denominators_include_failed_schema_attempts_and_costs():
    model = "openrouter:qwen/qwen4-max"
    common = {"model": model, "case_id": "synthetic", "elapsed_ms": 2}
    summary = summarize([{**common, "schema_valid": True, "truth_status": "within_tolerance",
                          "replay_status": "failed", "cost_estimate_usd": 0.2},
                         {**common, "schema_valid": False, "cost_estimate_usd": 0.1}])
    group = summary["groups"][f"{model}/synthetic"]
    assert group["denominator"] == 2 and group["schema_valid"] == 1
    assert group["truth"] == {"within_tolerance": 1, "unscored": 1}
    assert group["cost_estimate_usd"] == pytest.approx(0.3)


@pytest.mark.parametrize("status,stop", [(400, True), (401, True), (403, True),
                                       (404, True), (422, True), (429, False), (503, False)])
def test_only_stable_configuration_errors_stop_planned_reads(status, stop):
    error = LLMProviderError("synthetic")
    error.__cause__ = ModelHTTPError(status_code=status, model_name="synthetic")
    assert _stable_configuration_error(error) == stop


async def test_runner_stops_fixed_model_after_first_configuration_failure(monkeypatch, tmp_path):
    import argparse
    import hashlib
    import json

    from harness.platform import eval_runner, pokerdom_eval
    from tests.test_pokerdom_vision import table

    cases, out = tmp_path / "cases", tmp_path / "out"
    cases.mkdir()
    out.mkdir()
    image = tmp_path / "synthetic.jpg"
    image.write_bytes(b"synthetic image")
    (cases / "case.json").write_text(json.dumps({"case_id": "synthetic",
        "hero_nickname": "synthetic-alice", "reading": table().model_dump(mode="json"),
        "source": {"image_path": str(image), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}}))
    error = LLMProviderError("invalid configuration")
    error.__cause__ = ModelHTTPError(status_code=400, model_name="synthetic", body={"error": "synthetic"})
    fake_llm = AsyncMock(side_effect=error)
    monkeypatch.setattr(pokerdom_eval, "_private_output", lambda path: out)
    monkeypatch.setattr(pokerdom_eval.Config, "from_env", lambda: None)
    monkeypatch.setattr(pokerdom_eval, "LLM", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(pokerdom_eval, "read_pokerdom", fake_llm)
    monkeypatch.setattr(eval_runner, "_open_llm", AsyncMock(return_value=(None, 1, None)))
    monkeypatch.setattr(eval_runner, "_call_costs", AsyncMock(return_value=[]))
    monkeypatch.setattr(pokerdom_eval, "PROMPT", image)
    args = argparse.Namespace(cases=str(cases), out=str(out), model=["openrouter:qwen/qwen4-max"],
        case=None, offline=False, nickname="", terra_provider="openai", qwen_provider="alibaba", repeats=5)
    assert await pokerdom_eval.run(args) == 0
    assert fake_llm.await_count == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["planned_reads"] == 5 and summary["not_attempted"] == 4
    attempt = json.loads(next(out.glob("*.attempt.json")).read_text())
    assert attempt["configuration_stop"] is True
    assert attempt["cause_chain"][0]["body"] == {"error": "synthetic"}


def test_missing_cost_remains_unknown_instead_of_zero():
    model = "openrouter:qwen/qwen4-max"
    summary = summarize([{"model": model, "case_id": "synthetic", "elapsed_ms": 1,
                          "schema_valid": False}])
    group = summary["groups"][f"{model}/synthetic"]
    assert group["cost_estimate_usd"] is None and group["cost_unknown"] == 1
    assert group["provider_reported_cost_usd"] is None and group["provider_cost_unknown"] == 1


async def test_fixed_qwen_wire_schema_resolves_enum_key_refs_without_weakening(monkeypatch):
    from pydantic_ai import Agent

    from harness.contracts.pokerdom_vision import (
        PokerdomReading,
        PokerdomReadingV2,
        PokerdomTelegramReading,
    )
    from harness.contracts.raw import Street

    class NestedEnumMaps(BaseModel):
        values: dict[Street, list[Street]]
        optional: dict[Street, int] | None

    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    model = _build_model("openrouter:qwen/qwen4-max", fixed_provider="alibaba")
    assert isinstance(model, OpenRouterModel)
    captured = []

    async def capture(**kwargs):
        captured.append(kwargs["tools"][0]["function"]["parameters"])
        raise RuntimeError("offline capture only")

    with patch.object(model.client.chat.completions, "create", new=capture):
        for schema in (PokerdomReading, PokerdomReadingV2, NestedEnumMaps, PokerdomTelegramReading):
            with pytest.raises(RuntimeError, match="offline capture only"):
                await Agent(model, output_type=schema, retries=0).run("synthetic")

    def verify_refs(node, root):
        if isinstance(node, dict):
            if "$ref" in node:
                assert node["$ref"].startswith("#/")
                target = root
                for part in node["$ref"][2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            for value in node.values():
                verify_refs(value, root)
        elif isinstance(node, list):
            for value in node:
                verify_refs(value, root)

    for schema in captured:
        verify_refs(schema, schema)
    expected_streets = [street.value for street in Street]
    assert "format_observation" in captured[3]["required"]
    for schema in [*captured[:2], captured[3]]:
        for field in ("board", "pot_after_street"):
            assert schema["properties"][field]["propertyNames"]["enum"] == expected_streets
    assert captured[2]["properties"]["values"]["propertyNames"]["enum"] == expected_streets
    assert captured[2]["properties"]["values"]["additionalProperties"]["items"]["enum"] == expected_streets
    optional = captured[2]["properties"]["optional"]["anyOf"][0]
    assert optional["propertyNames"]["enum"] == expected_streets
    # The local observation contract still rejects non-street keys.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        NestedEnumMaps.model_validate({"values": {"unknown-street": []}, "optional": None})


async def test_reader_uses_v2_wire_not_legacy_ids(monkeypatch, tmp_path):
    from harness.contracts.pokerdom_vision import PokerdomReadingV2
    path = tmp_path / "v2.md"
    path.write_text("synthetic instructions")
    monkeypatch.setattr("harness.parsers.pokerdom_vision.PROMPT_PATH", path)
    llm = AsyncMock(return_value=(None, None))
    await read_pokerdom(llm, b"synthetic", trace_id=1, provider="alibaba", observer=None)
    assert llm.call_args.args[1] is PokerdomReadingV2


def test_cli_selects_exactly_two_remaining_cases_for_four_planned_reads():
    import argparse

    from harness.platform.pokerdom_eval import add_parser
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="suite"))
    args = parser.parse_args(["pokerdom-vision", "--case", "pd-002", "--case", "pd-003",
                              "--repeats", "1", "--out", "reports/synthetic"])
    assert args.case == ["pd-002", "pd-003"] and args.repeats == 1 and args.model is None
