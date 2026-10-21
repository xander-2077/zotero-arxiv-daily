"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser
import pytest

from zotero_arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    _parse_catchup_sections,
    _run_with_hard_timeout,
    _same_day_page_numbers,
)
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.delenv("TARGET_DATE", raising=False)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr(
        arxiv_retriever.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live RSS path must not call catchup")),
    )

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def _catchup_html(category, target_date, new_ids, cross_ids, rep_ids, extra=""):
    def items(ids):
        rows = []
        for index, paper_id in enumerate(ids, start=1):
            rows.append(
                f'<dt><a name="item{index}">[{index}]</a>'
                f'<a href="/abs/{paper_id}" title="Abstract" id="{paper_id}">arXiv:{paper_id}</a>'
                f'<a href="/pdf/{paper_id}" id="pdf-{paper_id}">pdf</a></dt>'
            )
        return "".join(rows)

    next_day = "2026-10-16"
    return f"""
    <html><head><title></title></head><body>
    <h1>Catchup results on {target_date}</h1>
    <a href="/catchup/{category}/{next_day}?abs=False&amp;page=1">Continue to the next day</a>
    {extra}
    <dl><h3>New submissions (showing {len(new_ids)} of {len(new_ids)} entries)</h3>{items(new_ids)}</dl>
    <dl><h3>Cross submissions (showing {len(cross_ids)} of {len(cross_ids)} entries)</h3>{items(cross_ids)}</dl>
    <dl><h3>Replacement submissions (showing {len(rep_ids)} of {len(rep_ids)} entries)</h3>{items(rep_ids)}</dl>
    </body></html>
    """


class _FakeResponse:
    def __init__(self, status_code, text, reason="OK"):
        self.status_code = status_code
        self.text = text
        self.reason = reason


def _install_catchup(monkeypatch, pages_by_category, searches):
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        category = url.rstrip("/").split("/")[-2]
        page = int((params or {}).get("page", "1"))
        body = pages_by_category[category][page]
        if isinstance(body, _FakeResponse):
            return body
        return _FakeResponse(200, body)

    class FakeSearch:
        def __init__(self, id_list=None, **kwargs):
            searches.append(list(id_list or []))

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            return iter(())

    monkeypatch.delenv("TARGET_DATE", raising=False)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(arxiv_retriever.requests, "get", _get)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Search", FakeSearch)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    return calls


def test_parse_catchup_sections_ignores_replacements_and_pdf_ids():
    html = _catchup_html("cs.AI", "2026-10-15", ["1111.11111"], ["2222.22222"], ["3333.33333"])
    sections = _parse_catchup_sections(html)
    assert sections == {
        "new": ["1111.11111"],
        "cross": ["2222.22222"],
        "replace": ["3333.33333"],
    }
    assert _same_day_page_numbers(html, "cs.AI", "2026-10-15") == set()
    assert _same_day_page_numbers(
        html + '<a href=/catchup/cs.AI/2026-10-15?abs=False&page=2>2</a>',
        "cs.AI",
        "2026-10-15",
    ) == {2}


def test_historical_listing_keeps_new_only_and_skips_rss(config, monkeypatch):
    config.source.arxiv.include_cross_list = False
    config.source.arxiv.target_date = "2026-10-15"
    searches = []
    calls = _install_catchup(
        monkeypatch,
        {
            "cs.AI": {1: _catchup_html("cs.AI", "2026-10-15", ["1111.11111", "4444.44444"], ["2222.22222"], ["3333.33333"])},
            "cs.CV": {1: _catchup_html("cs.CV", "2026-10-15", ["4444.44444"], ["5555.55555"], [])},
        },
        searches,
    )

    papers = ArxivRetriever(config)._retrieve_raw_papers()

    assert papers == []
    assert searches == [["1111.11111", "4444.44444"]]
    assert all("rss.arxiv.org" not in call["url"] for call in calls)
    assert [call["url"] for call in calls] == [
        "https://arxiv.org/catchup/cs.AI/2026-10-15",
        "https://arxiv.org/catchup/cs.CV/2026-10-15",
    ]


def test_historical_listing_includes_cross_lists_when_configured(config, monkeypatch):
    config.source.arxiv.include_cross_list = True
    config.source.arxiv.target_date = "2026-10-15"
    searches = []
    _install_catchup(
        monkeypatch,
        {
            "cs.AI": {1: _catchup_html("cs.AI", "2026-10-15", ["1111.11111"], ["2222.22222"], ["3333.33333"])},
            "cs.CV": {1: _catchup_html("cs.CV", "2026-10-15", ["4444.44444"], ["2222.22222"], [])},
        },
        searches,
    )

    ArxivRetriever(config)._retrieve_raw_papers()

    assert searches == [["1111.11111", "2222.22222", "4444.44444"]]


def test_historical_listing_follows_same_day_pages_only(config, monkeypatch):
    config.source.arxiv.category = ["cs.AI"]
    config.source.arxiv.target_date = "2026-10-15"
    page1 = _catchup_html(
        "cs.AI",
        "2026-10-15",
        ["1111.11111"],
        [],
        [],
        extra='<a href="/catchup/cs.AI/2026-10-15?abs=False&amp;page=2">2</a>',
    )
    page2 = _catchup_html("cs.AI", "2026-10-15", ["6666.66666"], [], [])
    searches = []
    calls = _install_catchup(monkeypatch, {"cs.AI": {1: page1, 2: page2}}, searches)

    ArxivRetriever(config)._retrieve_raw_papers()

    assert [call["params"]["page"] for call in calls] == ["1", "2"]
    assert searches == [["1111.11111", "6666.66666"]]


def test_historical_listing_empty_day_does_not_query_api(config, monkeypatch):
    config.source.arxiv.category = ["stat.ML"]
    config.source.arxiv.target_date = "2026-10-17"
    searches = []
    _install_catchup(
        monkeypatch,
        {"stat.ML": {1: "<html><head><title></title></head><body><p>No updates for Sat, 17 Oct 2026.</p></body></html>"}},
        searches,
    )

    assert ArxivRetriever(config)._retrieve_raw_papers() == []
    assert searches == []


def test_historical_listing_debug_truncates_before_api(config, monkeypatch):
    config.executor.debug = True
    config.source.arxiv.category = ["cs.AI"]
    config.source.arxiv.target_date = "2026-10-15"
    new_ids = [f"2610.{index:05d}" for index in range(1, 13)]
    searches = []
    _install_catchup(
        monkeypatch,
        {"cs.AI": {1: _catchup_html("cs.AI", "2026-10-15", new_ids, [], [])}},
        searches,
    )

    ArxivRetriever(config)._retrieve_raw_papers()

    assert searches == [new_ids[:10]]


def test_target_date_env_overrides_config(config, monkeypatch):
    config.source.arxiv.category = ["cs.AI"]
    config.source.arxiv.target_date = "2026-10-15"
    monkeypatch.setenv("TARGET_DATE", "2026-10-16")
    searches = []
    calls = _install_catchup(
        monkeypatch,
        {"cs.AI": {1: _catchup_html("cs.AI", "2026-10-16", ["7777.77777"], [], [])}},
        searches,
    )
    monkeypatch.setenv("TARGET_DATE", "2026-10-16")

    ArxivRetriever(config)._retrieve_raw_papers()

    assert calls[0]["url"].endswith("/2026-10-16")
    assert searches == [["7777.77777"]]


def test_invalid_target_date_raises_before_http(config, monkeypatch):
    config.source.arxiv.target_date = "2026-13-40"
    monkeypatch.delenv("TARGET_DATE", raising=False)
    monkeypatch.setattr(
        arxiv_retriever.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invalid date must not fetch")),
    )

    with pytest.raises(ValueError, match="real calendar date"):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_catchup_http_error_includes_arxiv_message(config, monkeypatch):
    config.source.arxiv.category = ["stat.ML"]
    config.source.arxiv.target_date = "2025-01-01"
    error = _FakeResponse(
        400,
        "<html><head><title>Invalid date: 2025-01-01. Catchup only allowed for past 90 days</title></head></html>",
        reason="Bad Request",
    )
    _install_catchup(monkeypatch, {"stat.ML": {1: error}}, [])

    with pytest.raises(RuntimeError, match="past 90 days"):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
