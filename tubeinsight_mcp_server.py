"""
TubeInsight MCP Server
Fetches YouTube comments and enables Claude to run sentiment analysis,
theme extraction, and full analytics reports.
"""

import json
import os
import re
from enum import Enum
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# ─── Constants ────────────────────────────────────────────────────────────────

YT_API_BASE = "https://www.googleapis.com/youtube/v3"
MAX_COMMENTS_PER_PAGE = 100
DEFAULT_COMMENT_LIMIT = 200

# ─── Server Init ──────────────────────────────────────────────────────────────

mcp = FastMCP("tubeinsight_mcp")
print("TubeInsight MCP Server initialized. Ready to fetch YouTube comments and generate insights!")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        raise ValueError(
            "YOUTUBE_API_KEY environment variable is not set. "
            "Get one at https://console.cloud.google.com/ and enable YouTube Data API v3."
        )
    return key


def _extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from various URL formats or return as-is."""
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",          # ?v=ID
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",   # youtu.be/ID
        r"(?:embed/)([a-zA-Z0-9_-]{11})",        # /embed/ID
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",       # /shorts/ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # Treat as raw video ID if it looks right
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id.strip()):
        return url_or_id.strip()
    raise ValueError(
        f"Could not extract a valid YouTube video ID from: '{url_or_id}'. "
        "Provide a full YouTube URL or an 11-character video ID."
    )


def _handle_yt_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 403:
            return (
                "Error 403: API key is invalid or YouTube Data API v3 is not enabled. "
                "Check your key at https://console.cloud.google.com/"
            )
        if status == 404:
            return "Error 404: Video not found. Check the URL or video ID."
        if status == 400:
            try:
                detail = e.response.json().get("error", {}).get("message", "")
                return f"Error 400: Bad request — {detail}"
            except Exception:
                return "Error 400: Bad request."
        if status == 429:
            return "Error 429: YouTube API quota exceeded. Try again tomorrow or use a different API key."
        return f"Error {status}: YouTube API request failed."
    if isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. YouTube API may be slow — try again."
    if isinstance(e, ValueError):
        return f"Error: {str(e)}"
    return f"Unexpected error: {type(e).__name__}: {str(e)}"


async def _fetch_comments_raw(
    video_id: str,
    limit: int,
    include_replies: bool,
    order: str,
) -> list[dict[str, Any]]:
    """Core paginated comment fetcher. Returns list of comment dicts."""
    api_key = _get_api_key()
    comments = []
    page_token = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(comments) < limit:
            params: dict[str, Any] = {
                "part": "snippet,replies" if include_replies else "snippet",
                "videoId": video_id,
                "maxResults": min(MAX_COMMENTS_PER_PAGE, limit - len(comments)),
                "order": order,
                "textFormat": "plainText",
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(f"{YT_API_BASE}/commentThreads", params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                entry: dict[str, Any] = {
                    "id": item["id"],
                    "author": top.get("authorDisplayName", "Anonymous"),
                    "text": top.get("textDisplay", ""),
                    "likes": top.get("likeCount", 0),
                    "published_at": top.get("publishedAt", ""),
                    "reply_count": item["snippet"].get("totalReplyCount", 0),
                }
                if include_replies and item["snippet"].get("totalReplyCount", 0) > 0:
                    replies = item.get("replies", {}).get("comments", [])
                    entry["replies"] = [
                        {
                            "author": r["snippet"].get("authorDisplayName", "Anonymous"),
                            "text": r["snippet"].get("textDisplay", ""),
                            "likes": r["snippet"].get("likeCount", 0),
                            "published_at": r["snippet"].get("publishedAt", ""),
                        }
                        for r in replies
                    ]
                comments.append(entry)

            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return comments


async def _fetch_video_metadata(video_id: str) -> dict[str, Any]:
    """Fetch basic video metadata (title, channel, view/like/comment counts)."""
    api_key = _get_api_key()
    async with httpx.AsyncClient(timeout=15.0) as client:
        params = {
            "part": "snippet,statistics",
            "id": video_id,
            "key": api_key,
        }
        resp = await client.get(f"{YT_API_BASE}/videos", params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("items"):
            raise ValueError(f"No video found for ID: {video_id}")
        item = data["items"][0]
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        return {
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "description_preview": snippet.get("description", "")[:300],
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
        }


def _format_comments_markdown(comments: list[dict[str, Any]], video_id: str) -> str:
    lines = [
        f"## YouTube Comments — video `{video_id}`",
        f"**Fetched:** {len(comments)} comments\n",
    ]
    for i, c in enumerate(comments, 1):
        lines.append(
            f"### [{i}] {c['author']} _(👍 {c['likes']} | 💬 {c['reply_count']} replies)_"
        )
        lines.append(c["text"])
        if c.get("replies"):
            for r in c["replies"]:
                lines.append(f"  > **↩ {r['author']}** (👍 {r['likes']}): {r['text']}")
        lines.append("")
    return "\n".join(lines)


# ─── Input Models ─────────────────────────────────────────────────────────────

class CommentOrder(str, Enum):
    RELEVANCE = "relevance"
    TIME = "time"


class FetchCommentsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_url: str = Field(
        ...,
        description="YouTube video URL or 11-character video ID. "
                    "E.g. 'https://youtu.be/dQw4w9WgXcQ' or 'dQw4w9WgXcQ'",
        min_length=3,
    )
    limit: int = Field(
        default=DEFAULT_COMMENT_LIMIT,
        description="Max number of top-level comments to fetch (1–500)",
        ge=1,
        le=500,
    )
    include_replies: bool = Field(
        default=False,
        description="Whether to include reply threads under each comment",
    )
    order: CommentOrder = Field(
        default=CommentOrder.RELEVANCE,
        description="Sort order: 'relevance' (most useful first) or 'time' (newest first)",
    )
    response_format: str = Field(
        default="markdown",
        description="Output format: 'markdown' for readable output or 'json' for structured data",
        pattern="^(markdown|json)$",
    )


class VideoMetadataInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_url: str = Field(
        ...,
        description="YouTube video URL or 11-character video ID",
        min_length=3,
    )


class SentimentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_url: str = Field(
        ...,
        description="YouTube video URL or 11-character video ID to analyze",
        min_length=3,
    )
    limit: int = Field(
        default=150,
        description="Number of comments to analyze for sentiment (1–300)",
        ge=1,
        le=300,
    )


class ReportInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_url: str = Field(
        ...,
        description="YouTube video URL or 11-character video ID",
        min_length=3,
    )
    limit: int = Field(
        default=200,
        description="Number of comments to include in the report (1–500)",
        ge=1,
        le=500,
    )


class TopCommentsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_url: str = Field(
        ...,
        description="YouTube video URL or 11-character video ID",
        min_length=3,
    )
    top_n: int = Field(
        default=10,
        description="Number of top comments to return ranked by likes",
        ge=1,
        le=50,
    )


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="tubeinsight_get_video_metadata",
    annotations={
        "title": "Get YouTube Video Metadata",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tubeinsight_get_video_metadata(params: VideoMetadataInput) -> str:
    """Fetch metadata for a YouTube video: title, channel, view count, like count, comment count.

    Use this before fetching comments to understand the video's context and scale.

    Args:
        params (VideoMetadataInput): Input containing:
            - video_url (str): YouTube URL or video ID

    Returns:
        str: Markdown-formatted video metadata with statistics
    """
    try:
        video_id = _extract_video_id(params.video_url)
        meta = await _fetch_video_metadata(video_id)
        return (
            f"## 📹 Video Metadata\n\n"
            f"**Title:** {meta['title']}\n"
            f"**Channel:** {meta['channel']}\n"
            f"**Published:** {meta['published_at'][:10]}\n"
            f"**Views:** {meta['view_count']:,}\n"
            f"**Likes:** {meta['like_count']:,}\n"
            f"**Comments:** {meta['comment_count']:,}\n\n"
            f"**Description preview:**\n{meta['description_preview']}..."
        )
    except Exception as e:
        return _handle_yt_error(e)


@mcp.tool(
    name="tubeinsight_get_comments",
    annotations={
        "title": "Fetch YouTube Comments",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tubeinsight_get_comments(params: FetchCommentsInput) -> str:
    """Fetch raw comments from any YouTube video URL or ID.

    Returns top-level comments (and optionally replies) sorted by relevance or time.
    Use this when you need the raw comment data to pass to other tools or for direct analysis.

    Args:
        params (FetchCommentsInput): Input containing:
            - video_url (str): YouTube URL or video ID
            - limit (int): Max comments to fetch (default 200, max 500)
            - include_replies (bool): Include reply threads (default False)
            - order (str): 'relevance' or 'time' (default 'relevance')
            - response_format (str): 'markdown' or 'json' (default 'markdown')

    Returns:
        str: Formatted list of comments with author, text, likes, and reply count
    """
    try:
        video_id = _extract_video_id(params.video_url)
        comments = await _fetch_comments_raw(
            video_id=video_id,
            limit=params.limit,
            include_replies=params.include_replies,
            order=params.order.value,
        )

        if not comments:
            return f"No comments found for video `{video_id}`. Comments may be disabled."

        if params.response_format == "json":
            return json.dumps({"video_id": video_id, "count": len(comments), "comments": comments}, indent=2)

        return _format_comments_markdown(comments, video_id)

    except Exception as e:
        return _handle_yt_error(e)


@mcp.tool(
    name="tubeinsight_get_top_comments",
    annotations={
        "title": "Get Top Comments by Likes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tubeinsight_get_top_comments(params: TopCommentsInput) -> str:
    """Fetch and rank the most-liked comments on a YouTube video.

    Ideal for quickly identifying the most resonant, popular, or viral comments
    without processing the full comment corpus.

    Args:
        params (TopCommentsInput): Input containing:
            - video_url (str): YouTube URL or video ID
            - top_n (int): How many top comments to return (default 10, max 50)

    Returns:
        str: Markdown-formatted ranked list of top comments by like count
    """
    try:
        video_id = _extract_video_id(params.video_url)
        # Fetch more to get a good pool for ranking
        comments = await _fetch_comments_raw(
            video_id=video_id,
            limit=min(params.top_n * 5, 300),
            include_replies=False,
            order="relevance",
        )
        if not comments:
            return f"No comments found for video `{video_id}`."

        ranked = sorted(comments, key=lambda c: c["likes"], reverse=True)[: params.top_n]

        lines = [f"## 🏆 Top {len(ranked)} Comments by Likes — `{video_id}`\n"]
        for i, c in enumerate(ranked, 1):
            lines.append(f"**#{i} — {c['author']}** (👍 {c['likes']})")
            lines.append(f"> {c['text']}\n")
        return "\n".join(lines)

    except Exception as e:
        return _handle_yt_error(e)


@mcp.tool(
    name="tubeinsight_sentiment_analysis",
    annotations={
        "title": "Analyse Comment Sentiment",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tubeinsight_sentiment_analysis(params: SentimentInput) -> str:
    """Fetch comments and return them structured for Claude to perform sentiment analysis.

    Prepares a prompt-ready payload with the comment corpus so the calling LLM
    can classify sentiment (positive / negative / neutral / mixed), identify emotional tones,
    and surface notable comment patterns.

    Args:
        params (SentimentInput): Input containing:
            - video_url (str): YouTube URL or video ID
            - limit (int): Number of comments to analyse (default 150, max 300)

    Returns:
        str: Structured comment corpus + analysis instructions for Claude
    """
    try:
        video_id = _extract_video_id(params.video_url)
        meta = await _fetch_video_metadata(video_id)
        comments = await _fetch_comments_raw(
            video_id=video_id,
            limit=params.limit,
            include_replies=False,
            order="relevance",
        )
        if not comments:
            return f"No comments found for video `{video_id}`. Cannot perform sentiment analysis."

        comment_corpus = "\n".join(
            f"[{i+1}] (👍{c['likes']}) {c['text']}" for i, c in enumerate(comments)
        )

        return (
            f"## 🎯 TubeInsight Sentiment Analysis Request\n\n"
            f"**Video:** {meta['title']}\n"
            f"**Channel:** {meta['channel']}\n"
            f"**Total comments fetched:** {len(comments)}\n\n"
            f"---\n\n"
            f"### Instructions for Analysis\n"
            f"Please analyse the following {len(comments)} YouTube comments and provide:\n\n"
            f"1. **Overall sentiment distribution** — percentage breakdown of positive / negative / neutral / mixed\n"
            f"2. **Dominant emotional tones** — e.g. excited, frustrated, nostalgic, critical, humorous\n"
            f"3. **Key themes** — top 5 topics/themes discussed in the comments\n"
            f"4. **Most positive comments** — 3 representative examples\n"
            f"5. **Most negative comments** — 3 representative examples\n"
            f"6. **Audience insights** — what does this comment section reveal about the audience?\n"
            f"7. **Creator action points** — what should the creator take note of?\n\n"
            f"---\n\n"
            f"### Comment Corpus\n\n"
            f"{comment_corpus}"
        )

    except Exception as e:
        return _handle_yt_error(e)


@mcp.tool(
    name="tubeinsight_full_report",
    annotations={
        "title": "Generate Full TubeInsight Analytics Report",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tubeinsight_full_report(params: ReportInput) -> str:
    """Fetch video metadata + comments and return a comprehensive report prompt for Claude.

    This is the all-in-one tool. It fetches everything — video stats, top comments,
    full comment corpus — and structures it as a prompt for Claude to generate a
    complete analytics report including sentiment, themes, audience profiling,
    engagement quality score, and creator recommendations.

    Args:
        params (ReportInput): Input containing:
            - video_url (str): YouTube URL or video ID
            - limit (int): Number of comments for the report (default 200, max 500)

    Returns:
        str: Full structured prompt with video data + comment corpus for Claude to analyse
    """
    try:
        video_id = _extract_video_id(params.video_url)
        meta = await _fetch_video_metadata(video_id)
        comments = await _fetch_comments_raw(
            video_id=video_id,
            limit=params.limit,
            include_replies=False,
            order="relevance",
        )
        if not comments:
            return f"No comments found for video `{video_id}`. Cannot generate report."

        top_10 = sorted(comments, key=lambda c: c["likes"], reverse=True)[:10]
        top_10_text = "\n".join(
            f"  [{i+1}] (👍{c['likes']}) {c['author']}: {c['text'][:200]}"
            for i, c in enumerate(top_10)
        )

        comment_corpus = "\n".join(
            f"[{i+1}] (👍{c['likes']}, 💬{c['reply_count']}) {c['text']}"
            for i, c in enumerate(comments)
        )

        engagement_rate = (
            ((meta["like_count"] + meta["comment_count"]) / meta["view_count"] * 100)
            if meta["view_count"] > 0
            else 0
        )

        return (
            f"## 📊 TubeInsight Full Analytics Report Request\n\n"
            f"### Video Stats\n"
            f"- **Title:** {meta['title']}\n"
            f"- **Channel:** {meta['channel']}\n"
            f"- **Published:** {meta['published_at'][:10]}\n"
            f"- **Views:** {meta['view_count']:,}\n"
            f"- **Likes:** {meta['like_count']:,}\n"
            f"- **Total Comments:** {meta['comment_count']:,}\n"
            f"- **Engagement Rate:** {engagement_rate:.2f}%\n"
            f"- **Comments analysed:** {len(comments)}\n\n"
            f"### Top 10 Comments by Likes\n{top_10_text}\n\n"
            f"---\n\n"
            f"### Report Instructions\n"
            f"Using the video stats and comment corpus below, generate a full TubeInsight Analytics Report with these sections:\n\n"
            f"**1. Executive Summary** (3–4 sentences)\n"
            f"**2. Sentiment Analysis** — overall tone breakdown with percentages\n"
            f"**3. Theme Analysis** — top 5–7 themes with frequency and representative quotes\n"
            f"**4. Audience Profile** — who is watching? Demographics, interests, intent\n"
            f"**5. Engagement Quality Score** — rate the engagement depth (superficial likes vs deep discussion)\n"
            f"**6. Controversy & Risk Flags** — any negative patterns, toxicity, or PR risks\n"
            f"**7. Creator Recommendations** — 5 actionable suggestions based on comments\n"
            f"**8. Viral Potential Signals** — comments suggesting share-worthy moments\n\n"
            f"---\n\n"
            f"### Full Comment Corpus ({len(comments)} comments)\n\n"
            f"{comment_corpus}"
        )

    except Exception as e:
        return _handle_yt_error(e)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()