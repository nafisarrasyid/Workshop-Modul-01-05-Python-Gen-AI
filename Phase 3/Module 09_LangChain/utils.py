# Phase 3 - GenAI Libraries: Module 09 (LangChain)
# Shared Utilities: Model Resolution, Compatibility Patches & Fast Offline Fallbacks

import os
import sys
import typing
import socket
import urllib.parse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ── Compatibility Patch 1: SQLite3 version for Chroma / LangChain ──────────
import sqlite3
if hasattr(sqlite3, "sqlite_version") and tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:2]) < (3, 35):
    sqlite3.sqlite_version = "3.35.0"
    sqlite3.sqlite_version_info = (3, 35, 0)

# ── Compatibility Patch 2: Python 3.9 Pydantic JSON Schema Fix ─────────────
try:
    import typing_extensions
    import pydantic.json_schema

    _orig_get_literal_values = pydantic.json_schema.get_literal_values

    def _safe_get_literal_values(annotation, **kwargs):
        for val in _orig_get_literal_values(annotation, **kwargs):
            if hasattr(val, "__args__") and (
                typing.get_origin(val) is typing.Literal
                or getattr(typing_extensions, "get_origin", lambda x: None)(val) is typing_extensions.Literal
            ):
                yield from _safe_get_literal_values(val, **kwargs)
            else:
                yield val

    pydantic.json_schema.get_literal_values = _safe_get_literal_values
except Exception:
    pass

# Load environment variables
env_paths = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent / "Module 08_Embeddings_SemanticSearch" / ".env",
    Path(__file__).parent.parent / "Module 06_LLM_APIs" / ".env",
]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def check_host_reachable(url: str, timeout: float = 0.5) -> bool:
    """Quick socket check to determine if the host and port are accepting connections."""
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class DeterministicMockEmbeddings(Embeddings):
    """
    Offline deterministic embedding model matching Module 08's mock_embed pattern.
    Produces repeatable vectors without network calls, suitable for offline testing.
    """
    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            words = set(text.lower().split())
            seed = sum(abs(hash(w)) % 10000 for w in words) % (2**31) if words else 42
            vec = np.random.default_rng(seed).standard_normal(self.dim)
            norm = np.linalg.norm(vec)
            vec = (vec / (norm if norm > 0 else 1.0)).tolist()
            results.append(vec)
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# Cache online status so we probe at most once per execution
_ONLINE_STATUS_CACHE = {}


def get_chat_model(
    provider: str = "openai",
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    mock_responses: list[str] = None,
) -> BaseChatModel:
    """
    Returns an initialized LangChain ChatModel.
    Automatically checks for:
      1. Direct provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
      2. OmniRoute / NineRouter local proxy (OMNIROUTE_API_KEY + OMNIROUTE_BASE_URL)
      3. Safe fallback to FakeListChatModel for offline testing if keys or network are unreachable.
    """
    default_responses = mock_responses or [
        "This is a simulated LangChain chat response for offline verification.",
        "LangChain Expression Language (LCEL) composes chains with the pipe operator.",
        "Retrieval-Augmented Generation connects language models to external data.",
    ]

    if provider.lower() == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OMNIROUTE_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OMNIROUTE_BASE_URL")
        model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

        if api_key:
            clean_url = base_url.rstrip("/") if base_url else None
            if clean_url and clean_url.endswith("/v1"):
                clean_url = clean_url[:-3]
            
            # Check reachability if using local proxy
            if clean_url and ("localhost" in clean_url or "127.0.0.1" in clean_url):
                if not check_host_reachable(clean_url):
                    return FakeListChatModel(responses=default_responses)

            try:
                return ChatAnthropic(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    base_url=clean_url if clean_url else None,
                    timeout=2,
                    max_retries=1,
                )
            except Exception:
                pass

        return FakeListChatModel(responses=default_responses)

    # Default to OpenAI-compatible
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return FakeListChatModel(responses=default_responses)

    # 2. Check Gemini direct OpenAI-compatible endpoint
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        gemini_url = os.getenv("GEMINI_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        gemini_model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        try:
            return ChatOpenAI(
                model=gemini_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=gemini_key,
                base_url=gemini_url,
                request_timeout=15,
                max_retries=1,
            )
        except Exception:
            pass

    # 3. Check OpenAI / OmniRoute
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OMNIROUTE_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OMNIROUTE_BASE_URL")
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    if api_key:
        clean_url = base_url.rstrip("/") if base_url else None
        if clean_url and not clean_url.endswith("/v1"):
            clean_url += "/v1"

        # Check reachability if using local proxy
        if clean_url and ("localhost" in clean_url or "127.0.0.1" in clean_url):
            if not check_host_reachable(clean_url):
                return FakeListChatModel(responses=default_responses)

        try:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=clean_url if clean_url else None,
                request_timeout=2,
                max_retries=1,
            )
        except Exception:
            pass

    return FakeListChatModel(responses=default_responses)


def get_embeddings() -> Embeddings:
    """
    Returns an initialized LangChain Embeddings instance.
    Uses OpenAIEmbeddings if API key and connection are available,
    otherwise returns DeterministicMockEmbeddings for offline reliability.
    """
    from langchain_openai import OpenAIEmbeddings

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OMNIROUTE_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OMNIROUTE_BASE_URL")

    if api_key:
        clean_url = base_url.rstrip("/") if base_url else None
        if clean_url and not clean_url.endswith("/v1"):
            clean_url += "/v1"

        if clean_url and ("localhost" in clean_url or "127.0.0.1" in clean_url):
            if not check_host_reachable(clean_url):
                return DeterministicMockEmbeddings(dim=32)

        try:
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
                base_url=clean_url if clean_url else None,
                request_timeout=2,
                max_retries=1,
            )
        except Exception:
            pass

    return DeterministicMockEmbeddings(dim=32)
