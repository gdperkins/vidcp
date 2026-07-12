"""Allow ``python -m vidcp`` (used by the MCP server to spawn background ingests)."""

from __future__ import annotations

from vidcp.cli import main

if __name__ == "__main__":
    main()
