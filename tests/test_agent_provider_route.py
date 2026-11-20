"""Agent-only routing: real facade and SDK against mock HTTP, never a model API."""
import json
from dataclasses import replace
from typing import Literal

import httpx2 as httpx
import pytest
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from harness.platform.config import InvalidEnvVar
from harness.platform.llm import LLM
from tests.test_llm_facade import Out, _make_trace_scope, fetch_all, replace_cfg


def test_config_agent_host_optional_normalized_and_vendor_bound(monkeypatch):
    cfg = replace_cfg(llm_agent_model='openrouter:synthetic/model')
    assert cfg.llm_agent_openrouter_provider == ''
    assert replace(cfg, llm_agent_openrouter_provider='  synthetic/host  ').llm_agent_openrouter_provider == 'synthetic/host'
    assert replace(cfg, llm_agent_model='anthropic:synthetic', llm_agent_openrouter_provider=' ').llm_agent_openrouter_provider == ''
    with pytest.raises(InvalidEnvVar, match='LLM_AGENT_OPENROUTER_PROVIDER'):
        replace(cfg, llm_agent_model='anthropic:synthetic', llm_agent_openrouter_provider='synthetic/host')
    monkeypatch.setenv('LLM_AGENT_OPENROUTER_PROVIDER', ' synthetic/host ')
    # Config.from_env convention exercised without depending on real process credentials.
    for name, value in {'LLM_AGENT_MODEL':'openrouter:synthetic/model','LLM_VERDICT_MODEL':'test','LLM_MAX_CONCURRENCY':'2','LLM_MAX_PER_MINUTE':'20','DATABASE_URL':'test','TELEGRAM_TOKEN':'test'}.items():
        monkeypatch.setenv(name, value)
    assert type(cfg).from_env().llm_agent_openrouter_provider == 'synthetic/host'


@pytest.mark.parametrize('fixed', [False, True])
async def test_agent_route_scope_profile_wire_and_sdk_retry(db_factory, monkeypatch, fixed):
    from harness.platform import llm as module
    requests = []
    sdk_retries = []
    profiles = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        name = body['tools'][0]['function']['name']
        return httpx.Response(200, json={
            'id':'synthetic','object':'chat.completion','created':1,'model':'synthetic/model',
            'provider':'SyntheticHost','choices':[{'index':0,'finish_reason':'tool_calls','message':{
                'role':'assistant','content':None,'tool_calls':[{'id':'call','type':'function','function':{
                    'name':name,'arguments':json.dumps({'text':'synthetic result'})}}]}}],
            'usage':{'prompt_tokens':4,'completion_tokens':3,'total_tokens':7}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        def provider(**kwargs):
            return OpenRouterProvider(**kwargs) if kwargs else OpenRouterProvider(api_key='test', http_client=client)
        def model(name, *, profile, settings, provider=None):
            provider = provider or OpenRouterProvider(api_key='test', http_client=client)
            sdk_retries.append(provider.client.max_retries)
            result = OpenRouterModel(name, provider=provider, profile=profile, settings=settings)
            profiles.append(result.profile)
            return result
        monkeypatch.setattr(module, 'OpenRouterProvider', provider)
        monkeypatch.setattr(module, 'OpenRouterModel', model)
        cfg = replace_cfg(llm_agent_model='openrouter:synthetic/model', llm_vision_model='openrouter:synthetic/model',
            llm_openrouter_provider_order=('global/host',), llm_agent_openrouter_provider='synthetic/host' if fixed else '')
        llm = LLM(cfg, db_factory)
        trace = await _make_trace_scope(db_factory)
        purposes: tuple[Literal['agent','vision_extract'], ...] = ('agent','vision_extract')
        for purpose in purposes:
            result, _ = await llm(purpose, Out, prompt='synthetic', trace_id=trace)
            assert result.text == 'synthetic result'
    baseline = {'sort':'throughput','require_parameters':True,'allow_fallbacks':True,'order':['global/host']}
    assert requests[1]['provider'] == baseline
    assert requests[0]['provider'] == (baseline | {'only':['synthetic/host'],'allow_fallbacks':False} if fixed else baseline)
    before, after = dict(requests[0]), dict(requests[1])
    before.pop('provider'); after.pop('provider')
    assert before == after  # tools, forced choice, strict, reasoning, model preserved
    assert profiles[0] == profiles[1]
    assert sdk_retries == ([0,2] if fixed else [2,2])
    assert await fetch_all(db_factory, 'select status, served_by from llm_calls order by id') == [('ok','SyntheticHost'),('ok','SyntheticHost')]


async def test_fixed_agent_http_errors_have_no_hidden_sdk_retries(db_factory, monkeypatch):
    from harness.platform import llm as module
    from harness.platform.llm import LLMProviderError
    count = 0
    def respond(request):
        nonlocal count
        count += 1
        return httpx.Response(503, json={'error':{'message':'synthetic unavailable'}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        def provider(**kwargs):
            return OpenRouterProvider(**kwargs) if kwargs else OpenRouterProvider(api_key='test',http_client=client)
        monkeypatch.setattr(module,'OpenRouterProvider',provider)
        monkeypatch.setattr(module,'_backoff_delay',lambda _:0)
        llm = LLM(replace_cfg(llm_agent_model='openrouter:synthetic/model',llm_agent_openrouter_provider='synthetic/host'),db_factory)
        with pytest.raises(LLMProviderError):
            await llm('agent',Out,prompt='synthetic',trace_id=await _make_trace_scope(db_factory))
    assert count == module._MAX_HTTP_ATTEMPTS  # facade policy retained, SDK adds no retries
    assert await fetch_all(db_factory,'select status from llm_calls') == [('error',)] * count
