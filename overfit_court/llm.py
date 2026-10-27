"""Nebius Token Factory client: Nano/Super/Ultra routing, live-model fallbacks and a cost ledger.

Token Factory exposes an OpenAI-compatible API, so this uses the openai SDK
pointed at a different base URL. Without NEBIUS_API_KEY it runs in demo mode,
so the whole pipeline can be developed, tested and demoed offline.

Nemotron 3 models think before they answer unless told not to. Each role
says whether it wants that (`reasoning`): planning, writing and repairing code,
fishing and explaining the verdict do; auditing code and triaging search
results don't, so they stay quick and cheap. The switch is the model's chat
template flag, `enable_thinking`.

Live models misbehave in predictable ways, and each is handled here instead
of crashing a trial:

- the configured model id differs from the account's only in case or prefix:
  the id the account lists is used;
- a request with optional parts (JSON mode, the thinking switch) is rejected:
  retry without each in turn, then without both. The part whose removal made
  it work is never sent to that model again; JSON mode gives way to asking for
  JSON in the prompt;
- a reasoning model spends its whole token budget thinking, or its answer is
  cut off: retry once with four times the budget;
- reasoning arrives inline as <think>...</think>, or unclosed: stripped;
- the reply is not valid JSON: ask once more, quoting the parse error;
- rate limits, timeouts and server errors: the SDK retries with backoff.

Every request, retries and failures included, is logged in `calls` with its
tokens, latency and cost.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from openai import BadRequestError, OpenAI, UnprocessableEntityError

from .config import Settings

REASONING = re.compile(r"<think>.*?</think>", re.DOTALL)
UNFINISHED_REASONING = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL)
MAX_TOKENS_CAP = 32_000
JSON_ONLY = "\n\nReply with a single JSON object and nothing else."
JSON_MODE, REASONING_SWITCH = "JSON mode", "the thinking switch"  # optional request parts


class LLMError(RuntimeError):
    """A model returned nothing usable, even after retrying."""


@dataclass
class Call:
    """One request to Token Factory (or one demo-mode reply)."""

    role: str  # planner, coder, auditor, clerk, judge, fisherman...
    tier: str  # nano, super or ultra
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    note: str = ""  # why this request was a retry, or why it failed


class LLM:
    """One client per trial: routes each role to its Nemotron tier and logs every call.

    With no NEBIUS_API_KEY and no injected `client` it runs in demo mode: each
    call returns the `mock` reply its caller supplies, so the pipeline still runs.
    """

    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.mock = settings.mock_llm and client is None
        self.calls: list[Call] = []
        self.refused: dict[str, set[str]] = {}  # model -> optional request parts it rejected
        self._listed: list[str] | None = None  # model ids the account can use, once fetched
        self.client = client
        if not self.mock and self.client is None:
            # The SDK itself retries rate limits, timeouts and server errors with backoff.
            self.client = OpenAI(
                base_url=settings.base_url,
                api_key=settings.nebius_api_key,
                max_retries=4,
                timeout=180.0,
            )

    def chat(
        self,
        role: str,
        tier: str,
        system: str,
        user: str,
        *,
        mock: Callable[[], str],
        json_mode: bool = False,
        reasoning: bool | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """Ask the `tier` model; return its answer with any reasoning stripped.

        `reasoning` turns the model's thinking on or off; None leaves its default.
        """
        model = self.model_for(tier)
        if self.mock:
            self.calls.append(Call(role, tier, f"{model} (mock)", 0, 0, 0.0, 0.0))
            return mock()

        refused = self.refused.setdefault(model, set())
        wanted = {}  # the optional request parts this model hasn't refused
        if reasoning is not None and REASONING_SWITCH not in refused:
            wanted[REASONING_SWITCH] = reasoning
        if json_mode and JSON_MODE not in refused:
            wanted[JSON_MODE] = True

        def ask(parts: dict, budget: int, note: str = ""):
            prompt = system + (JSON_ONLY if json_mode and JSON_MODE not in parts else "")
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
            return self._send(
                role=role,
                tier=tier,
                model=model,
                messages=messages,
                max_tokens=budget,
                optional=parts,
                note=note,
            )

        # If a request is rejected, perhaps the model refuses one of its optional parts:
        # try without each in turn, then without all. The model is marked as refusing
        # what a retry left out only if that retry succeeds; if every retry fails, the
        # request itself was the problem and its error is raised.
        tries = [wanted] + [{k: v for k, v in wanted.items() if k != part} for part in wanted]
        if len(wanted) > 1:
            tries.append({})
        for n, parts in enumerate(tries):
            left_out = [part for part in wanted if part not in parts]
            note = "retry without " + " and ".join(left_out) if left_out else ""
            try:
                raw, finish_reason = ask(parts, max_tokens, note)
                break
            except Exception as e:
                if not (_is_bad_request(e) and n + 1 < len(tries)):
                    raise
        refused.update(left_out)

        answer = strip_reasoning(raw)
        if (not answer or finish_reason == "length") and max_tokens < MAX_TOKENS_CAP:
            # Usually a reasoning model that spent the budget thinking, or an answer cut
            # off mid-way (truncated code would only waste a repair).
            bigger = min(MAX_TOKENS_CAP, max_tokens * 4)
            note = f"retry with max_tokens={bigger} ({finish_reason})"
            raw, finish_reason = ask(parts, bigger, note)
            answer = strip_reasoning(raw)
        if not answer:
            raise LLMError(
                f"{model} returned no answer for '{role}' (finish_reason={finish_reason}). "
                "Try a larger max_tokens or a different MODEL_* setting."
            )
        return answer

    def chat_json(self, role: str, tier: str, system: str, user: str, *, mock, **options) -> dict:
        """Like chat(), but returns a JSON object; asks once more if the reply doesn't parse."""
        reply = self.chat(role, tier, system, user, mock=mock, json_mode=True, **options)
        try:
            return parse_json(reply)
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            if self.mock:
                raise
            again = (
                f"{user}\n\nYour previous reply could not be parsed as JSON ({e}). "
                "Reply with a single valid JSON object and nothing else."
            )
            return parse_json(
                self.chat(role, tier, system, again, mock=mock, json_mode=True, **options)
            )

    def cost_table(self) -> list[dict]:
        return [asdict(call) for call in self.calls]

    def total_cost(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    def model_for(self, tier: str) -> str:
        """The model id for a tier, as the account lists it if it differs only in case or prefix."""
        configured = self.settings.models[tier]
        if self.mock:
            return configured
        if self._listed is None:
            try:
                self._listed = [m.id for m in self.client.models.list().data]
            except Exception:  # listing is only a convenience; the configured id may still work
                self._listed = []
        if configured in self._listed:
            return configured
        wanted = configured.lower().split("/")[-1]
        matches = [listed for listed in self._listed if listed.lower().split("/")[-1] == wanted]
        return matches[0] if matches else configured

    def _send(
        self,
        *,
        role: str,
        tier: str,
        model: str,
        messages: list,
        max_tokens: int,
        optional: dict,
        note: str = "",
    ) -> tuple[str, str | None]:
        """One request to Token Factory. Returns (raw content, finish reason) and logs the call.

        Thinking roles sample as NVIDIA's examples for Nemotron do (temperature
        0.6, top_p 0.95); the rest keep a low temperature for steady answers. A
        request that fails is logged too (with no tokens), then the error is raised.
        """
        thinking = optional.get(REASONING_SWITCH, True)
        options = {"temperature": 0.6, "top_p": 0.95} if thinking else {"temperature": 0.2}
        if JSON_MODE in optional:
            options["response_format"] = {"type": "json_object"}
        if REASONING_SWITCH in optional:
            options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, **options
            )
        except Exception as e:
            failure = f"failed: {type(e).__name__}: {e}"[:300]
            latency = round(time.perf_counter() - start, 3)
            note = f"{note}; {failure}" if note else failure
            self.calls.append(Call(role, tier, model, 0, 0, latency, 0.0, note))
            raise
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        price_in, price_out = self.settings.token_prices[tier]  # USD per million tokens
        self.calls.append(
            Call(
                role=role,
                tier=tier,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=round(time.perf_counter() - start, 3),
                cost_usd=(input_tokens * price_in + output_tokens * price_out) / 1e6,
                note=note,
            )
        )
        choice = response.choices[0]
        return choice.message.content or "", getattr(choice, "finish_reason", None)


def _is_bad_request(e: Exception) -> bool:
    """Did the API reject the request itself (rather than fail to answer it)?"""
    if isinstance(e, BadRequestError | UnprocessableEntityError):
        return True
    return getattr(e, "status_code", None) in (400, 422)


def strip_reasoning(text: str | None) -> str:
    """Remove <think> blocks, including one the model never closed."""
    text = REASONING.sub("", text or "")
    return UNFINISHED_REASONING.sub("", text).strip()


def parse_json(text: str) -> dict:
    """The JSON object in a reply, tolerating reasoning, code fences and prose around it."""
    text = strip_reasoning(text)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        embedded = re.search(r"\{.*\}", text, re.DOTALL)
        if not embedded:
            raise
        parsed = json.loads(embedded.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def extract_code(text: str) -> str:
    """The Python in a reply: the last fenced block defining positions(), else the last block."""
    text = strip_reasoning(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    defining = [block for block in blocks if "def positions" in block]
    return (defining[-1] if defining else blocks[-1]).strip() + "\n"
