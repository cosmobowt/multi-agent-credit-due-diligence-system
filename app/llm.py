"""Provider-agnostic model factory with built-in rate limiting.

`init_chat_model` lets you swap providers/models with one env var (LLM_MODEL).
LLM_RPM controls the client-side pace so a full pipeline run (research loop +
risk + writer + reviewer + any revise loop) stays under the provider's
requests-per-minute cap instead of bursting into 429 errors.

One *shared* limiter across all agents paces every call from the same bucket,
since the quota is per-project, not per-call-site.
"""
import os

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

# Requests/minute to self-impose. Set LLM_RPM in .env to match your model's
# free-tier cap. Default 6 stays safely under tight free-tier caps (some
# accounts are limited to 10 RPM even on flash-lite). Raise it on a paid tier.
_RPM = float(os.getenv("LLM_RPM", "6"))

_rate_limiter = InMemoryRateLimiter(
    requests_per_second=_RPM / 60.0,
    check_every_n_seconds=0.1,
    max_bucket_size=1,  # no bursts; strict pacing
)


def get_llm(temperature: float = 0.2):
    model = os.getenv("LLM_MODEL", "google_genai:gemini-2.5-flash")
    return init_chat_model(
        model,
        temperature=temperature,
        rate_limiter=_rate_limiter,
        max_retries=3,
    )