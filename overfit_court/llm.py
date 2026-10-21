"""Nebius Token Factory client with Nano/Super/Ultra routing and a cost ledger.

Token Factory exposes an OpenAI-compatible API, so we use the openai SDK with
a different base_url. Without NEBIUS_API_KEY the client runs in mock mode so
the whole pipeline can be developed, tested and demoed offline.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable

from .config import Settings

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Call:
    role: str
    tier: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_llm
        self.ledger: list[Call] = []
        self.client = None
        if not self.mock:
            from openai import OpenAI
            self.client = OpenAI(base_url=settings.base_url, api_key=settings.nebius_api_key)

    def chat(self, role: str, tier: str, system: str, user: str, *,
             mock: Callable[[], str], json_mode: bool = False,
             max_tokens: int = 2048, temperature: float = 0.2) -> str:
        model = self.s.models[tier]
        t0 = time.perf_counter()
        if self.mock:
            text = mock()
            self.ledger.append(Call(role, tier, model + " (mock)", 0, 0, 0.0, 0.0))
            return text
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}])
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        u = resp.usage
        tin = getattr(u, "prompt_tokens", 0) or 0
        tout = getattr(u, "completion_tokens", 0) or 0
        pin, pout = self.s.prices[tier]
        cost = tin * pin / 1e6 + tout * pout / 1e6
        self.ledger.append(Call(role, tier, model, tin, tout, round(latency, 3), cost))
        return THINK_RE.sub("", resp.choices[0].message.content or "").strip()

    def chat_json(self, role, tier, system, user, *, mock, **kw) -> dict:
        text = self.chat(role, tier, system, user, mock=mock, json_mode=True, **kw)
        return parse_json(text)

    def cost_table(self) -> list[dict]:
        return [asdict(c) for c in self.ledger]

    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.ledger)


def parse_json(text: str) -> dict:
    text = THINK_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip() + "\n"
