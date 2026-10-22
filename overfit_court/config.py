"""Settings, read from environment variables (or a .env file in the working directory).

Nothing here is required: with no keys at all the app runs in demo mode.
See .env.example for every variable. A value that is set but can't be used
(an unknown executor, a cost that isn't a number or is negative) raises an
error rather than being quietly replaced by a default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

from dotenv import load_dotenv

load_dotenv()

EXECUTOR_NAMES = ("nebius", "local")  # Token Factory Sandboxes, or subprocesses (development)


def _env_number(name: str, default, kind=float, minimum=0):
    """The environment variable `name` as a finite number of type `kind`, at least `minimum`.

    `default` if it is unset. A negative cost, say, would pay strategies to trade.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid {kind.__name__}") from None
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name}={raw!r} must be a finite number of at least {minimum}")
    return value


@dataclass
class Settings:
    nebius_api_key: str | None
    base_url: str  # Token Factory's OpenAI-compatible endpoint
    tavily_api_key: str | None
    models: dict = field(default_factory=dict)  # tier ("nano", "super", "ultra") -> model id
    token_prices: dict = field(default_factory=dict)  # tier -> (USD per 1M input, 1M output)
    executor: str = "local"  # "nebius" (Token Factory Sandboxes) or "local" (subprocesses)
    max_configs: int = 120  # largest parameter grid tried in one trial
    cost_bps: float = 5.0  # trading cost per unit of turnover, in basis points

    @property
    def mock_llm(self) -> bool:
        """Demo mode for the models: no key, so agents use library-based replies."""
        return not self.nebius_api_key

    @property
    def mock_search(self) -> bool:
        return not self.tavily_api_key

    def offline(self) -> Settings:
        """A copy with no API keys: demo mode, fully reproducible, nothing leaves the machine."""
        return replace(self, nebius_api_key=None, tavily_api_key=None)


def get_settings() -> Settings:
    # With a key, model-written code runs, so it belongs in Token Factory Sandboxes
    # unless OC_EXECUTOR explicitly says otherwise.
    default_executor = "nebius" if os.getenv("NEBIUS_API_KEY") else "local"
    executor = (os.getenv("OC_EXECUTOR") or default_executor).strip().lower()
    if executor not in EXECUTOR_NAMES:
        raise ValueError(
            f"OC_EXECUTOR={executor!r}: use 'nebius' (Token Factory Sandboxes) or 'local' "
            "(subprocesses on this machine, for development)."
        )
    # One configuration alone can't be tested for luck.
    max_configs = _env_number("OC_MAX_CONFIGS", 120, int, minimum=2)
    return Settings(
        nebius_api_key=os.getenv("NEBIUS_API_KEY"),
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        # Model ids differ between accounts: `python -m overfit_court.preflight` lists yours.
        models={
            "nano": os.getenv("MODEL_NANO", "nvidia/nemotron-3-nano-30b-a3b"),
            "super": os.getenv("MODEL_SUPER", "nvidia/nemotron-3-super-120b-a12b"),
            "ultra": os.getenv("MODEL_ULTRA", "nvidia/nemotron-3-ultra-550b-a55b"),
        },
        # Only used to report spend; check your Token Factory console for current prices.
        token_prices={
            "nano": (_env_number("PRICE_NANO_IN", 0.06), _env_number("PRICE_NANO_OUT", 0.24)),
            "super": (_env_number("PRICE_SUPER_IN", 0.30), _env_number("PRICE_SUPER_OUT", 0.90)),
            "ultra": (_env_number("PRICE_ULTRA_IN", 1.00), _env_number("PRICE_ULTRA_OUT", 3.00)),
        },
        executor=executor,
        max_configs=max_configs,
        cost_bps=_env_number("OC_COST_BPS", 5.0),
    )
