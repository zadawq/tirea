# Travel UI

CopilotKit + AG-UI travel-planning demo with a map canvas, trip sidebar, frontend actions, and approval-gated trip creation.

## Architecture

```text
Browser (CopilotKit) -> Next.js runtime (/api/copilotkit) -> Rust backend (travel agent) -> tools/state
                      -> frontend action: highlight_place
                      -> HITL action: add_trips
```

The frontend uses `HttpAgent` against `POST /v1/ag-ui/agents/travel/runs`.

## Quick Start

Terminal 1:

```bash
DEEPSEEK_API_KEY=<your-key> cargo run -p tirea-examples --bin travel
```

Terminal 2:

```bash
cd examples/travel-ui
npm install
npm run dev
```

Open `http://localhost:3000`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8080` | Backend base URL used by the CopilotKit runtime |

Agent id is fixed to `travel` in `app/layout.tsx`.

## Verify

1. Prompt `Plan a 3-day Tokyo trip with food and museums.`
2. Confirm an approval card appears for `add_trips`.
3. Approve the action and verify the sidebar updates with the new trip.
4. Ask the agent to highlight a place and verify the map recenters through the frontend action.

## Key Files

- `app/page.tsx`
- `lib/copilotkit-app.ts`
- `hooks/useTripActions.ts`
- `hooks/useTripApproval.tsx`
- `../src/travel.rs`
- `../src/travel/tools.rs`

## Related Docs

- [Integrate CopilotKit (AG-UI)](../../docs/book/src/how-to/integrate-copilotkit-ag-ui.md)
- [Enable Tool Permission HITL](../../docs/book/src/how-to/enable-tool-permission-hitl.md)
- [Expose HTTP SSE](../../docs/book/src/how-to/expose-http-sse.md)
