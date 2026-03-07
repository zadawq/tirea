# Research UI

CopilotKit + AG-UI research workflow demo with a resource panel, report canvas, browser-opening frontend actions, and approval-gated resource deletion.

## Architecture

```text
Browser (CopilotKit) -> Next.js runtime (/api/copilotkit) -> Rust backend (research agent) -> tools/state
                      -> frontend action: open_resource
                      -> HITL action: delete_resources
```

The frontend uses `HttpAgent` against `POST /v1/ag-ui/agents/research/runs`.

## Quick Start

Terminal 1:

```bash
DEEPSEEK_API_KEY=<your-key> cargo run -p tirea-examples --bin research
```

Terminal 2:

```bash
cd examples/research-ui
npm install
npm run dev
```

Open `http://localhost:3001`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8080` | Backend base URL used by the CopilotKit runtime |

Agent id is fixed to `research` in `app/layout.tsx`.

## Verify

1. Prompt `Research the current EV battery supply chain and build a short report.`
2. Verify the research question, resources, and report content update across the three panels.
3. Ask the agent to open a source and verify a new browser tab is opened through the frontend action.
4. Ask the agent to delete resources and verify the approval card appears before deletion proceeds.

## Key Files

- `app/page.tsx`
- `lib/copilotkit-app.ts`
- `hooks/useResearchActions.ts`
- `hooks/useDeleteApproval.tsx`
- `../src/research.rs`
- `../src/research/tools.rs`

## Related Docs

- [Integrate CopilotKit (AG-UI)](../../docs/book/src/how-to/integrate-copilotkit-ag-ui.md)
- [Enable Tool Permission HITL](../../docs/book/src/how-to/enable-tool-permission-hitl.md)
- [Expose HTTP SSE](../../docs/book/src/how-to/expose-http-sse.md)
