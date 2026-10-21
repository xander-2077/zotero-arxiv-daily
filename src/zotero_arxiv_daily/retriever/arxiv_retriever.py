from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar, resolve_arxiv_target_date
from tempfile import TemporaryDirectory
import feedparser
from html import unescape
from html.parser import HTMLParser
from tqdm import tqdm
import multiprocessing
import os
import re
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from urllib.parse import quote
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
CATCHUP_PAGE_LIMIT = 10
_CATCHUP_USER_AGENT = "zotero-arxiv-daily/1.0 (historical announcement backfill)"
_TITLE_RE = re.compile(r"<title>(.*?)</title>", flags=re.IGNORECASE | re.DOTALL)
_SHOWING_FIRST_RE = re.compile(r"showing first \d+ of \d+", flags=re.IGNORECASE)


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


class _CatchupHTMLParser(HTMLParser):
    """Split a catchup page into new, cross, and replacement arXiv ids."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.section: str | None = None
        self.ids: dict[str, list[str]] = {"new": [], "cross": [], "replace": []}
        self._in_h3 = False
        self._h3: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value for key, value in attrs}
        if tag == "h3":
            self._in_h3 = True
            self._h3 = []
            return
        if tag != "a" or self.section not in self.ids:
            return
        href = attr.get("href") or ""
        paper_id = attr.get("id")
        if paper_id and href.startswith("/abs/"):
            self.ids[self.section].append(paper_id)

    def handle_endtag(self, tag: str) -> None:
        if tag != "h3" or not self._in_h3:
            return
        heading = "".join(self._h3).strip().lower()
        if heading.startswith("new submission"):
            self.section = "new"
        elif heading.startswith("cross"):
            self.section = "cross"
        elif heading.startswith("replacement"):
            self.section = "replace"
        else:
            self.section = None
        self._in_h3 = False

    def handle_data(self, data: str) -> None:
        if self._in_h3:
            self._h3.append(data)


def _parse_catchup_sections(html: str) -> dict[str, list[str]]:
    parser = _CatchupHTMLParser()
    parser.feed(html)
    return parser.ids


def _html_title(html: str) -> str:
    match = _TITLE_RE.search(html)
    if match is None:
        return ""
    return re.sub(r"\s+", " ", unescape(match.group(1))).strip()


def _same_day_page_numbers(html: str, category: str, target_date: str) -> set[int]:
    needle = f"/catchup/{quote(category, safe='')}/{target_date}"
    pages: set[int] = set()
    hrefs = re.finditer(
        r"""href\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""",
        html,
    )
    for match in hrefs:
        href = unescape(next(group for group in match.groups() if group))
        if needle not in href:
            continue
        page_match = re.search(r"[?&]page=(\d+)", href)
        if page_match:
            pages.add(int(page_match.group(1)))
    return pages


def _dedupe_ids(paper_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for paper_id in paper_ids:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        ordered.append(paper_id)
    return ordered


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        target_date = resolve_arxiv_target_date(self.config)
        if target_date:
            # RSS is current-only. Catchup is arXiv's historical /new listing.
            all_paper_ids = self._collect_historical_paper_ids(target_date)
        else:
            all_paper_ids = self._collect_rss_paper_ids()
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]
        return self._fetch_papers_by_ids(all_paper_ids)

    def _collect_rss_paper_ids(self) -> list[str]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        return [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]

    def _collect_historical_paper_ids(self, target_date: str) -> list[str]:
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        categories = list(self.config.source.arxiv.category)
        logger.info(
            f"Fetching arXiv catchup listing for {target_date} "
            f"({', '.join(categories)}; include_cross_list={include_cross_list})"
        )
        paper_ids: list[str] = []
        for index, category in enumerate(categories):
            if index:
                sleep(1)
            sections = self._fetch_catchup_sections(category, target_date)
            selected = list(sections["new"])
            if include_cross_list:
                selected.extend(sections["cross"])
            logger.info(
                f"{category} {target_date}: {len(sections['new'])} new, "
                f"{len(sections['cross'])} cross, {len(sections['replace'])} replacements"
            )
            paper_ids.extend(selected)
        paper_ids = _dedupe_ids(paper_ids)
        logger.info(f"Historical arXiv listing {target_date}: {len(paper_ids)} papers")
        return paper_ids

    def _fetch_catchup_sections(self, category: str, target_date: str) -> dict[str, list[str]]:
        combined = {"new": [], "cross": [], "replace": []}
        seen_pages: set[int] = set()
        pending = [1]
        while pending:
            page = pending.pop(0)
            if page in seen_pages:
                continue
            if len(seen_pages) >= CATCHUP_PAGE_LIMIT:
                logger.warning(f"Stopped reading {category} {target_date} after {CATCHUP_PAGE_LIMIT} catchup pages")
                break
            seen_pages.add(page)
            html = self._get_catchup_html(category, target_date, page)
            if page == 1 and _SHOWING_FIRST_RE.search(html) and _same_day_page_numbers(html, category, target_date) <= {1}:
                logger.warning(
                    f"arXiv catchup page for {category} on {target_date} looks truncated "
                    "and has no further page link"
                )
            for section, ids in _parse_catchup_sections(html).items():
                combined[section].extend(ids)
            for next_page in sorted(_same_day_page_numbers(html, category, target_date)):
                if next_page not in seen_pages:
                    pending.append(next_page)
        return combined

    def _get_catchup_html(self, category: str, target_date: str, page: int) -> str:
        url = f"https://arxiv.org/catchup/{quote(category, safe='')}/{target_date}"
        headers = {"User-Agent": _CATCHUP_USER_AGENT}
        max_attempts = 3
        for attempt in range(max_attempts):
            response = requests.get(
                url,
                params={"abs": "False", "page": str(page)},
                headers=headers,
                timeout=DOWNLOAD_TIMEOUT,
            )
            if response.status_code == 200:
                return response.text
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                wait = 5 * (attempt + 1)
                logger.warning(
                    f"arXiv catchup HTTP {response.status_code} for {category} {target_date} "
                    f"page {page}, retry {attempt + 1}/{max_attempts} in {wait}s"
                )
                sleep(wait)
                continue
            title = _html_title(response.text)
            detail = title or response.reason or "unknown error"
            raise RuntimeError(
                f"arXiv catchup listing for {category} on {target_date} failed "
                f"(HTTP {response.status_code}): {detail}"
            )
        raise RuntimeError(f"arXiv catchup listing for {category} on {target_date} failed")

    def _fetch_papers_by_ids(self, all_paper_ids: list[str]) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        raw_papers = []
        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 5
        batch_retry_delay = 30
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = batch_retry_delay * (attempt + 1)
                        logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    else:
                        raise
            if i + 20 < len(all_paper_ids):
                sleep(3)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
