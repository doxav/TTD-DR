"""Tests for SearXNG integration into the TTD-DR pipeline."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph_ttd_dr.utils import validate_search_engines
from langgraph_ttd_dr.tools import WebSearchTool
from langgraph_ttd_dr.state import create_initial_state
from langgraph_ttd_dr.interface import TTDResearcher
import langgraph_ttd_dr.nodes as nodes_module


# ---------- Unit tests ----------


def test_validate_search_engines_accepts_searxng():
    engines = validate_search_engines(["tavily", "searxng", "invalid"])
    assert "tavily" in engines
    assert "searxng" in engines
    assert "invalid" not in engines


def test_web_search_tool_uses_searxng_when_configured(monkeypatch):
    """WebSearchTool should initialize and query SearXNG when configured."""
    monkeypatch.setenv("SEARXNG_API_URL", "http://searxng.test/search")
    monkeypatch.setenv("SEARXNG_API_KEY", "dummy")

    import langgraph_ttd_dr.tools as tools_module

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, timeout=None):
        if (params or {}).get("q") == "healthcheck":
            return DummyResponse({"results": []})
        return DummyResponse(
            {
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1", "content": "Snippet 1"},
                    {"title": "Result 2", "url": "https://example.com/2", "content": "Snippet 2"},
                ]
            }
        )

    monkeypatch.setattr(tools_module, "requests", types.SimpleNamespace(get=fake_get))

    tool = WebSearchTool()
    results = tool.search("test", max_results=2, enabled_engines=["searxng"])

    assert len(results) == 2
    assert all(res["source"] == "searxng" for res in results)
    status = tool.get_status()
    assert status["searxng"] is True
    assert status["any_available"] is True


# ---------- Tests on higher-level components ----------


def test_ttd_researcher_accepts_searxng_engine():
    dummy_client = object()
    researcher = TTDResearcher(
        client=dummy_client,
        search_engines=["searxng"],
        max_iterations=1,
        max_sources=1,
    )
    assert "searxng" in researcher.search_engines


def test_search_agent_node_uses_state_search_engines(monkeypatch):
    from langgraph_ttd_dr.nodes import SearchAgentNode

    calls = []

    def fake_search_web(query, max_results=5, enabled_engines=None):
        calls.append(
            {
                "query": query,
                "max_results": max_results,
                "enabled_engines": list(enabled_engines or []),
            }
        )
        engine = (enabled_engines or ["mock"])[0]
        return [
            {
                "title": "Stub result",
                "url": "https://example.com",
                "content": "stub content",
                "score": 1.0,
                "source": engine,
            }
        ]

    monkeypatch.setattr(nodes_module, "search_web", fake_search_web)

    node = SearchAgentNode(client=None)
    monkeypatch.setattr(
        node,
        "_process_search_results_with_evolution",
        lambda *args, **kwargs: "Evolved answer from fake evolution",
    )

    state = create_initial_state(
        original_query="Test query",
        search_engines=["searxng"],
    )
    state["search_questions"] = ["What is SearXNG?"]

    update = node(state)

    assert calls, "search_web was never called"
    assert all("searxng" in call["enabled_engines"] for call in calls)
    assert "search_results" in update
    assert "search_answers" in update
    assert len(update["search_answers"]) == len(state["search_questions"])
    assert update["search_answers"][0]["answer"].startswith("Evolved answer")

