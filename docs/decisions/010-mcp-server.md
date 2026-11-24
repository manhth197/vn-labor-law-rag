# 010. MCP server over stdio: search, article lookup and answers as tools

Date: 2026-11-24 · Status: accepted (optional M6 item proposed by the coordinator)

## Context
Agent clients (Claude Code, Claude Desktop, IDE assistants) use tools through the Model Context Protocol. The project already has the pieces such a tool needs: a retriever with scores, an article index with amendment notes, and an answer graph with verified citations. The goal is to expose them without a second code path that could drift from the API.

## Decision
- **Official SDK, stdio transport:** `mcp` 2.3 (`MCPServer`), started with `python -m vnlaw_rag.cli mcp`. A local client spawns the server, so there is no port to open and nothing to authenticate. The SDK sits in an optional extra `[mcp]`; the API and the Docker image do not need it.
- **Three tools, with different costs:**
  - `search_labor_law` (passages with reference, score, official link and amendment notes; no LLM call);
  - `get_labor_law_article` (one article);
  - `ask_labor_law` (the full graph, which calls Gemini).

  An agent can do its own reasoning over the search results and call the LLM tool only when it needs a cited answer.
- **Same components as the API:** the tools call the same retriever, `build_graph` and `load_articles`, so citation checks, the `grade` threshold, warnings and the corpus notice are identical.
- **Typed inputs and outputs:**
  - Inputs are checked by Pydantic: question length 1-500, `top_k` 1-10, `doc` restricted to `BLLD2019`/`ND145`.
  - Outputs are Pydantic models, so clients get structured content.
  - Anticipated failures (unknown article, Gemini rate limit or overload) raise `ToolError` with the API's Vietnamese message.
- **Lazy loading:** bge-m3 and the database pool open on the first tool call that needs them, under a lock (sync tools run on worker threads), and close with the server. Listing tools and reading an article load nothing heavy. Nothing prints to stdout, which carries the protocol.

## Alternatives
- Streamable HTTP transport: useful for remote clients, but it needs authentication and rate limiting; not needed for a local tool.
- Only an `ask` tool: simpler, but every call would spend a Gemini request, and agents could not inspect the sources.

## Consequences
- **Tests:** an in-memory MCP client lists and calls the tools with a fake retriever and a fake LLM; they cover the schemas, the dropped fake citation, the friendly rate-limit error and argument validation. CI installs `.[dev,mcp]`.
- **stdio check (2026-11-24):** the three tools are listed, the Điều 112 lookup returns the 01/03/2027 note, and the first `search_labor_law` returned Điều 25 at 0.741 in 45.5 s, including the bge-m3 load.
