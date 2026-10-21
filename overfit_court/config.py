"""Settings, all overridable from environment variables or a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    nebius_api_key: str | None
    base_url: str
    tavily_api_key: str | None
    models: dict = field(default_factory=dict)
    # USD per million tokens (input, output). Check your Token Factory console.
    prices: dict = field(default_factory=dict)
    executor: str = "local"
    max_configs: int = 120
    cost_bps: float = 5.0

    @property
    def mock_llm(self) -> bool:
        return not self.nebius_api_key

    @property
    def mock_search(self) -> bool:
        return not self.tavily_api_key


def get_settings() -> Settings:
    return Settings(
        nebius_api_key=os.getenv("NEBIUS_API_KEY"),
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        # Verify exact IDs with GET /v1/models on your account (python -m overfit_court.preflight).
        models={
            "nano": os.getenv("MODEL_NANO", "nvidia/nemotron-3-nano-30b-a3b"),
            "super": os.getenv("MODEL_SUPER", "nvidia/nemotron-3-super-120b-a12b"),
            "ultra": os.getenv("MODEL_ULTRA", "nvidia/nemotron-3-ultra-550b-a55b"),
        },
        prices={
            "nano": (_f("PRICE_NANO_IN", 0.06), _f("PRICE_NANO_OUT", 0.24)),
            "super": (_f("PRICE_SUPER_IN", 0.30), _f("PRICE_SUPER_OUT", 0.90)),
            "ultra": (_f("PRICE_ULTRA_IN", 1.00), _f("PRICE_ULTRA_OUT", 3.00)),
        },
        executor=os.getenv("OC_EXECUTOR", "local"),
        max_configs=int(os.getenv("OC_MAX_CONFIGS", 120)),
        cost_bps=_f("OC_COST_BPS", 5.0),
    )
