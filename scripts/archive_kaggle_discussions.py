"""Archive every discussion topic for a Kaggle competition.

The archive contains human-readable Markdown and machine-readable JSON.  It uses
the official Kaggle CLI package/API and follows all topic/comment pagination.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bleach
from kaggle.api.kaggle_api_extended import KaggleApi
from slugify import slugify


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


DEFAULT_COMPETITION = "rogii-wellbore-geology-prediction"
DEFAULT_OUTPUT = Path("docs/discussion")
TOPICS_PER_PAGE = 20  # Fixed by the competition-topics API.

ALLOWED_TAGS = {
    "a",
    "b",
    "blockquote",
    "br",
    "code",
    "del",
    "details",
    "div",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "img",
    "kbd",
    "li",
    "ol",
    "p",
    "pre",
    "s",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "u",
    "ul",
}
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "img": ["alt", "height", "src", "title", "width"],
    "code": ["class"],
    "span": ["class"],
    "div": ["class"],
}


@dataclass(frozen=True)
class TopicSummary:
    id: int
    title: str
    author_name: str
    comment_count: int
    votes: int
    post_date: str

    @classmethod
    def from_api(cls, topic: Any) -> "TopicSummary":
        return cls(
            id=int(getattr(topic, "id")),
            title=str(getattr(topic, "title", "") or ""),
            author_name=str(getattr(topic, "author_name", "") or ""),
            comment_count=int(getattr(topic, "comment_count", 0) or 0),
            votes=int(getattr(topic, "votes", 0) or 0),
            post_date=str(getattr(topic, "post_date", "") or ""),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing raw JSON file instead of downloading that topic again.",
    )
    return parser.parse_args()


def make_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def fetch_all_topic_summaries(api: KaggleApi, competition: str) -> list[TopicSummary]:
    topics_by_id: dict[int, TopicSummary] = {}
    page = 1
    total_count: int | None = None

    while True:
        response = api.competition_list_topics(competition, sort_by="new", page=page)
        page_topics = list(response.topics or [])
        if total_count is None:
            value = getattr(response, "total_count", None)
            total_count = int(value) if value is not None else None

        for topic in page_topics:
            summary = TopicSummary.from_api(topic)
            topics_by_id[summary.id] = summary

        total_label = str(total_count) if total_count is not None else "?"
        print(
            f"Topic list page {page}: {len(page_topics)} items "
            f"({len(topics_by_id)}/{total_label} unique)",
            flush=True,
        )

        if not page_topics:
            break
        if total_count is not None and len(topics_by_id) >= total_count:
            break
        if total_count is None and len(page_topics) < TOPICS_PER_PAGE:
            break
        page += 1

    return sorted(topics_by_id.values(), key=lambda item: (item.post_date, item.id), reverse=True)


def comment_to_dict(comment: Any) -> dict[str, Any]:
    replies = [comment_to_dict(reply) for reply in (getattr(comment, "replies", None) or [])]
    return {
        "id": int(getattr(comment, "id", 0) or 0),
        "authorName": str(getattr(comment, "author_name", "") or ""),
        "postDate": str(getattr(comment, "post_date", "") or ""),
        "votes": int(getattr(comment, "votes", 0) or 0),
        "content": str(getattr(comment, "content", "") or ""),
        "replies": replies,
    }


def topic_to_dict(topic: Any, comments: list[Any], competition: str) -> dict[str, Any]:
    topic_id = int(getattr(topic, "id"))
    return {
        "id": topic_id,
        "title": str(getattr(topic, "title", "") or ""),
        "authorName": str(getattr(topic, "author_name", "") or ""),
        "postDate": str(getattr(topic, "post_date", "") or ""),
        "votes": int(getattr(topic, "votes", 0) or 0),
        "commentCount": int(getattr(topic, "comment_count", 0) or 0),
        "content": str(getattr(topic, "content", "") or ""),
        "url": f"https://www.kaggle.com/competitions/{competition}/discussion/{topic_id}",
        "comments": [comment_to_dict(comment) for comment in comments],
    }


_thread_local = threading.local()


def worker_api() -> KaggleApi:
    api = getattr(_thread_local, "api", None)
    if api is None:
        api = make_api()
        _thread_local.api = api
    return api


def fetch_topic(summary: TopicSummary, competition: str) -> dict[str, Any]:
    topic, comments, next_page_token = worker_api().forums_topic_show(summary.id)
    if next_page_token:
        raise RuntimeError(f"Topic {summary.id} still has an unconsumed comment page token")
    if topic is None:
        raise RuntimeError(f"Topic {summary.id} was not found")
    return topic_to_dict(topic, list(comments or []), competition)


def clean_content(content: str) -> str:
    if not content:
        return "_本文なし_"
    cleaned = bleach.clean(
        content,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols={"http", "https", "mailto"},
        strip=True,
    ).strip()
    return html.unescape(cleaned) if cleaned else "_本文なし_"


def count_nested_comments(comments: list[dict[str, Any]]) -> int:
    return sum(1 + count_nested_comments(comment.get("replies", [])) for comment in comments)


def render_comment(comment: dict[str, Any], path: tuple[int, ...], depth: int) -> list[str]:
    label = ".".join(str(part) for part in path)
    heading_level = min(3 + depth, 6)
    heading = "#" * heading_level
    author = comment.get("authorName") or "Unknown"
    lines = [
        f"{heading} コメント {label} — {author}",
        "",
        f"- 投稿日時: {comment.get('postDate', '')}",
        f"- 投票数: {comment.get('votes', 0)}",
        f"- コメントID: `{comment.get('id', 0)}`",
        "",
        clean_content(str(comment.get("content", ""))),
        "",
    ]
    for index, reply in enumerate(comment.get("replies", []), start=1):
        lines.extend(render_comment(reply, (*path, index), depth + 1))
    return lines


def render_topic_markdown(data: dict[str, Any]) -> str:
    comments = data.get("comments", [])
    actual_comment_count = count_nested_comments(comments)
    lines = [
        f"# {data['title']}",
        "",
        f"- 投稿者: {data.get('authorName') or 'Unknown'}",
        f"- 投稿日時: {data.get('postDate', '')}",
        f"- 投票数: {data.get('votes', 0)}",
        f"- コメント数: {data.get('commentCount', 0)}（取得数: {actual_comment_count}）",
        f"- トピックID: `{data['id']}`",
        f"- 原文: [{data['url']}]({data['url']})",
        "",
        "## 本文",
        "",
        clean_content(str(data.get("content", ""))),
        "",
        "## コメント",
        "",
    ]
    if not comments:
        lines.extend(["_コメントなし_", ""])
    else:
        for index, comment in enumerate(comments, start=1):
            lines.extend(render_comment(comment, (index,), 0))
    return "\n".join(lines).rstrip() + "\n"


def topic_filename(topic: dict[str, Any]) -> str:
    title_slug = slugify(topic.get("title", ""), max_length=80) or "topic"
    return f"{topic['id']}_{title_slug}.md"


def write_topic(output: Path, data: dict[str, Any]) -> str:
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = topic_filename(data)
    (output / filename).write_text(render_topic_markdown(data), encoding="utf-8")
    (raw_dir / f"{data['id']}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return filename


def render_index(
    competition: str,
    topics: list[dict[str, Any]],
    filenames: dict[int, str],
    archived_at: str,
) -> str:
    total_comments = sum(count_nested_comments(topic.get("comments", [])) for topic in topics)
    lines = [
        "# ROGIIコンペティション Discussionアーカイブ",
        "",
        f"- コンペティション: `{competition}`",
        f"- 取得日時（UTC）: {archived_at}",
        f"- トピック数: {len(topics)}",
        f"- 取得コメント・返信数: {total_comments}",
        "- 取得元: Kaggle公式CLI/API",
        "",
        "各トピックの閲覧用Markdownと、原文データを保持する `raw/*.json` を保存しています。",
        "",
        "| 投稿日時 | トピック | 投稿者 | 投票 | コメント |",
        "|---|---|---|---:|---:|",
    ]
    for topic in topics:
        topic_id = int(topic["id"])
        title = str(topic.get("title", "")).replace("|", "\\|")
        author = str(topic.get("authorName") or "Unknown").replace("|", "\\|")
        lines.append(
            f"| {topic.get('postDate', '')} | [{title}]({filenames[topic_id]}) "
            f"| {author} | {topic.get('votes', 0)} | "
            f"{count_nested_comments(topic.get('comments', []))} |"
        )
    lines.extend(
        [
            "",
            "---",
            "",
            "このアーカイブは取得時点の内容です。編集・削除・追加を含む最新状態はKaggle上で確認してください。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        list_api = make_api()
        summaries = fetch_all_topic_summaries(list_api, args.competition)
    except Exception as exc:
        print(f"Failed to list Kaggle topics: {exc}", file=sys.stderr)
        return 1

    print(f"Fetching {len(summaries)} discussion topics with {args.workers} workers", flush=True)
    topics_by_id: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}

    pending: list[TopicSummary] = []
    for summary in summaries:
        raw_path = raw_dir / f"{summary.id}.json"
        if args.resume and raw_path.exists():
            try:
                topics_by_id[summary.id] = json.loads(raw_path.read_text(encoding="utf-8"))
                print(f"Reused topic {summary.id}: {summary.title}", flush=True)
                continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append(summary)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_topic, item, args.competition): item for item in pending}
        completed = len(topics_by_id)
        for future in as_completed(futures):
            summary = futures[future]
            try:
                data = future.result()
                topics_by_id[summary.id] = data
                write_topic(output, data)
                completed += 1
                print(
                    f"[{completed}/{len(summaries)}] Saved {summary.id}: {summary.title}",
                    flush=True,
                )
            except Exception as exc:
                failures[summary.id] = str(exc)
                print(f"FAILED {summary.id}: {summary.title}: {exc}", file=sys.stderr, flush=True)

    ordered_topics = [topics_by_id[item.id] for item in summaries if item.id in topics_by_id]
    filenames: dict[int, str] = {}
    for data in ordered_topics:
        filenames[int(data["id"])] = write_topic(output, data)

    archived_at = datetime.now(timezone.utc).isoformat()
    (output / "README.md").write_text(
        render_index(args.competition, ordered_topics, filenames, archived_at),
        encoding="utf-8",
    )
    manifest = {
        "competition": args.competition,
        "archivedAt": archived_at,
        "expectedTopicCount": len(summaries),
        "archivedTopicCount": len(ordered_topics),
        "failedTopics": failures,
        "topics": [
            {
                "id": data["id"],
                "title": data["title"],
                "file": filenames[int(data["id"])],
                "rawFile": f"raw/{data['id']}.json",
                "commentCount": count_nested_comments(data.get("comments", [])),
            }
            for data in ordered_topics
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Archive complete: {len(ordered_topics)}/{len(summaries)} topics at {output}",
        flush=True,
    )
    if failures:
        print("Rerun with --resume to retry only missing topics.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
