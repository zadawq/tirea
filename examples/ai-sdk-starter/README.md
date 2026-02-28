# AI SDK Starter

Vite + React + React Router frontend using Vercel AI SDK v6 (`@ai-sdk/react`) to chat with a tirea agent backend.

## Architecture

```
Browser (useChat) → Rust Agent (axum + CORS) → LLM
                    (SSE direct)
```

The Rust agent at `agent/` serves AI SDK v6 UI Message Stream events as SSE. The Vite SPA connects directly via CORS — no Node.js proxy layer.

Three demo pages:
- `/` — Canvas with todo board, shared state panel, theme toggle
- `/basic` — Backend tools (weather, stock, notes), tool approval, interactive dialogs
- `/threads` — Backend-persisted thread management with history

## Prerequisites

- Node.js 18+
- Rust toolchain (for the agent binary)
- `DEEPSEEK_API_KEY` environment variable

## Quick Start

```bash
cd examples/ai-sdk-starter

# Install frontend dependencies
npm install

# Start both frontend (Vite) and backend (Rust agent) concurrently
DEEPSEEK_API_KEY=<key> npm run dev
```

Or start them separately:

```bash
# Terminal 1: Rust agent
DEEPSEEK_API_KEY=<key> cargo run -p ai-sdk-starter-agent

# Terminal 2: Vite frontend
cd examples/ai-sdk-starter
npm run dev:ui
```

Open http://localhost:3001 and send a message.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `VITE_BACKEND_URL` | `http://localhost:38080` | Rust agent address (frontend) |
| `AGENTOS_HTTP_ADDR` | `127.0.0.1:38080` | Rust agent listen address |
| `AGENT_MODEL` | `deepseek-chat` | LLM model ID |
| `AGENT_MAX_ROUNDS` | `8` | Max tool/model loop iterations |

## Verify

1. Open http://localhost:3001
2. Type a message (e.g. "What's the weather in Tokyo?")
3. Confirm streaming response with weather card appears
4. Navigate between Canvas / Basic / Threads pages
