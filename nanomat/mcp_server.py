"""NanoMatAI as a tool for language models: a Model Context Protocol server.

Any MCP client - Claude Code or Desktop, Cursor, or an agent loop around a local
model - can spawn this over stdio and answer a question in words ("2D
semiconductors near 1.5 eV without lead") from the model's numbers, each with the
model's own verdict attached. The tools themselves live in `nanomat/llm_tools.py`
as plain functions, so tests can call them without a protocol in between.

    python -m nanomat.mcp_server        # stdio; the client spawns it

    claude mcp add nanomat -- /path/to/venv/bin/python -m nanomat.mcp_server

Needs the shipped weights and `screening_table.csv` at the repository root
(`scripts/precompute_screening.py` writes it). Nothing is printed to stdout but
the protocol itself.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

try:  # mcp 2.x renamed FastMCP to MCPServer; the decorator surface is unchanged
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from nanomat import llm_tools

mcp = _Server("nanomat")
for _fn in (llm_tools.predict_structure, llm_tools.find_structures, llm_tools.search_materials,
            llm_tools.get_material, llm_tools.model_card):
    mcp.tool()(_fn)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
