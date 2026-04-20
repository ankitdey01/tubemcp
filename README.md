# TubeInsight MCP Server

An MCP server that connects Claude to the YouTube Data API v3 — fetch raw comments from any YouTube video and let Claude run sentiment analysis, theme extraction, and full analytics reports.

## Tools

| Tool | Description |
|------|-------------|
| `tubeinsight_get_video_metadata` | Fetch title, channel, views, likes, comment count |
| `tubeinsight_get_comments` | Fetch raw comments (paginated, up to 500) |
| `tubeinsight_get_top_comments` | Rank comments by likes |
| `tubeinsight_sentiment_analysis` | Structured prompt for Claude sentiment analysis |
| `tubeinsight_full_report` | Full analytics report prompt — sentiment + themes + audience + recommendations |

## Setup

### 1. Get a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **YouTube Data API v3**
3. Create an API Key under **Credentials**

### 2. Install Dependencies

```bash
pip install mcp httpx pydantic
```

### 3. Set Environment Variable

```bash
export YOUTUBE_API_KEY=your_key_here
```

### 4. Run the Server

```bash
python server.py
```

### 5. Connect to Claude Code

Add to your `claude_desktop_config.json` or `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "tubeinsight": {
      "command": "python",
      "args": ["/path/to/tubeinsight_mcp/server.py"],
      "env": {
        "YOUTUBE_API_KEY": "your_key_here"
      }
    }
  }
}
```

## Example Usage

Once connected to Claude, you can say:

- *"Analyse the comments on this video: https://youtube.com/watch?v=..."*
- *"What are people saying about [video]? Give me a full sentiment report."*
- *"What are the top 10 most liked comments on this video?"*
- *"Give me a full TubeInsight analytics report for this YouTube link."*

## API Quota

YouTube Data API v3 has a **10,000 unit daily quota** (free tier).
Each `commentThreads` page costs **1 unit**. Fetching 200 comments = ~2 units.
A full report with 500 comments = ~5 units. Well within free limits for normal usage.

## Project Structure

```
tubeinsight_mcp/
├── server.py      # MCP server (single file)
└── README.md
```
