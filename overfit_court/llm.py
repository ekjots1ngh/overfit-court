"""Nebius Token Factory client with Nano/Super/Ultra routing and a cost ledger.

Token Factory exposes an OpenAI-compatible API, so we use the openai SDK with
a different base_url. Without NEBIUS_API_KEY the client runs in mock mode so
the whole pipeline can be developed, tested and demoed offline.

Live models misbehave in predictable ways, and each is handled here rather
than crashing a trial:
- a model or deployment rejects response_format (JSON mode): retry once
  without it, ask for JSON in the prompt, and remember not to send it again
- a reasoning model spends its whole token budget thinking and returns empty
  or unfinished content: retry once with a larger budget
- reasoning arrives inline as <think>...</think>, or unclosed: stripped
- the reply is not valid JSON: re-ask once, quoting the parse error
- rate limits, timeouts and 5xx: the SDK retries with exponential backoff
Every request that reaches the API, including retries, is logged with tokens,
latency and cost.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable

from .config import Settings

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
UNCLOSED_THINK_RE = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL)
MAX_TOKENS_CAP = 16000


class LLMError(RuntimeError):
    pass


@dataclass
class Call:
    role: str
    tier: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    note: str = ""


def strip_reasoning(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    return UNCLOSED_THINK_RE.sub("", text).strip()


class LLM:
    def __init__(self, settings: Settings, client=None):
        self.s = settings
        self.mock = settings.mock_llm and client is None
        self.ledger: list[Call] = []
        self._no_json_mode: set[str] = set()
        self.client = client
        if not self.mock and self.client is None:
            from openai import OpenAI
            self.client = OpenAI(base_url=settings.base_url, api_key=settings.nebius_api_key,
                                 max_retries=4, timeout=180.0)

    # ------------------------------------------------------------ one request
    def _request(self, role, tier, model, messages, max_tokens, temperature, json_mode, note):
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=messages)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        u = getattr(resp, "usage", None)
        tin = getattr(u, "prompt_tokens", 0) or 0
        tout = getattr(u, "completion_tokens", 0) or 0
        pin, pout = self.s.prices[tier]
        self.ledger.append(Call(role, tier, model, tin, tout, round(latency, 3),
                                tin * pin / 1e6 + tout * pout / 1e6, note))
        choice = resp.choices[0]
        return choice.message.content or "", getattr(choice, "finish_reason", None)

    # ------------------------------------------------------------ public API
    def chat(self, role: str, tier: str, system: str, user: str, *,
             mock: Callable[[], str], json_mode: bool = False,
             max_tokens: int = 2048, temperature: float = 0.2) -> str:
        model = self.s.models[tier]
        if self.mock:
            text = mock()
            self.ledger.append(Call(role, tier, model + " (mock)", 0, 0, 0.0, 0.0))
            return text

        use_json = json_mode and model not in self._no_json_mode
        if json_mode and not use_json:
            system += "\n\nReply with a single JSON object and nothing else."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            raw, finish = self._request(role, tier, model, messages, max_tokens, temperature,
                                        use_json, "")
        except Exception as e:
            if not (use_json and _is_bad_request(e)):
                raise
            # this model or deployment doesn't accept response_format: ask in the prompt instead
            self._no_json_mode.add(model)
            messages[0]["content"] = system + "\n\nReply with a single JSON object and nothing else."
            raw, finish = self._request(role, tier, model, messages, max_tokens, temperature,
                                        False, "retry without JSON mode")
            use_json = False

        text = strip_reasoning(raw)
        if not text and max_tokens < MAX_TOKENS_CAP:
            # typically a reasoning model that used the whole budget thinking
            bigger = min(MAX_TOKENS_CAP, max_tokens * 4)
            raw, finish = self._request(role, tier, model, messages, bigger, temperature,
                                        use_json, f"retry with max_tokens={bigger} ({finish})")
            text = strip_reasoning(raw)
        if not text:
            raise LLMError(f"{model} returned no answer for '{role}' (finish_reason={finish}). "
                           "Try a larger max_tokens or a different MODEL_* setting.")
        return text

    def chat_json(self, role, tier, system, user, *, mock, **kw) -> dict:
        text = self.chat(role, tier, system, user, mock=mock, json_mode=True, **kw)
        try:
            return parse_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            if self.mock:
                raise
            retry_user = (f"{user}\n\nYour previous reply could not be parsed as JSON ({e}). "
                          "Reply with a single valid JSON object and nothing else.")
            return parse_json(self.chat(role, tier, system, retry_user, mock=mock,
                                        json_mode=True, **kw))

    def cost_table(self) -> list[dict]:
        return [asdict(c) for c in self.ledger]

    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.ledger)


def _is_bad_request(e: Exception) -> bool:
    try:
        from openai import BadRequestError, UnprocessableEntityError
        if isinstance(e, (BadRequestError, UnprocessableEntityError)):
            return True
    except ImportError:  # pragma: no cover
        pass
    return getattr(e, "status_code", None) in (400, 422)


def parse_json(text: str) -> dict:
    text = strip_reasoning(text)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError(f"expected a JSON object, got {type(out).__name__}")
    return out


def extract_code(text: str) -> str:
    text = strip_reasoning(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        # prefer the block that actually defines positions()
        with_fn = [b for b in blocks if "def positions" in b]
        return (with_fn[-1] if with_fn else blocks[-1]).strip() + "\n"
    return text.strip() + "\n"
