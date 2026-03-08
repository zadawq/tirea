# Ecosystem Integrations

This page summarizes common AG-UI and AI SDK integration patterns and maps them to this project.

## Which Frontend Path To Choose

Use this quick rule:

| Need | Preferred path |
|---|---|
| Fastest chat-style integration with `useChat` | AI SDK v6 |
| Rich shared state, frontend tools, generative UI, HITL | AG-UI / CopilotKit |
| Backend-only tools with a thin frontend adapter | AI SDK v6 |
| Frontend-executed tools that suspend and resume runs | AG-UI / CopilotKit |
| Canvas-style or co-agent-like UX | AG-UI / CopilotKit |

Repo-specific mapping:

| Integration | Backend endpoint | Frontend runtime shape | Best for |
|---|---|---|---|
| AI SDK v6 | `POST /v1/ai-sdk/agents/:agent_id/runs` | `useChat` + SSE adapter route | plain chat, minimal glue, backend-centric tools |
| AG-UI / CopilotKit | `POST /v1/ag-ui/agents/:agent_id/runs` | CopilotKit runtime proxy + `HttpAgent` | shared state, frontend tools, approvals, canvas UX |

## In-Repo Shortest Paths

- AI SDK: `examples/ai-sdk-starter/`
- CopilotKit / AG-UI: `examples/copilotkit-starter/`

These are the primary references when you need full working frontend integrations rather than protocol reference alone.

## CopilotKit Integration Patterns

### Pattern A: Runtime Proxy + `HttpAgent` (remote backend)

Used when agent runtime is hosted separately and exposes AG-UI over HTTP.

- CopilotKit runtime runs in Next.js route (`/api/copilotkit`).
- `HttpAgent` points to backend AG-UI run endpoint.

Examples:

- [with-langgraph-fastapi](https://github.com/CopilotKit/with-langgraph-fastapi)
- [with-microsoft-agent-framework-python](https://github.com/CopilotKit/with-microsoft-agent-framework-python)
- [with-microsoft-agent-framework-dotnet](https://github.com/CopilotKit/with-microsoft-agent-framework-dotnet)

### Pattern B: Runtime Proxy + framework-specific adapter

Used when framework has a dedicated CopilotKit adapter.

- `LangGraphHttpAgent` / `LangGraphAgent`
- `MastraAgent`

Examples:

- [with-langgraph-js](https://github.com/CopilotKit/with-langgraph-js)
- [with-mastra](https://github.com/CopilotKit/with-mastra)

### Pattern C: Canvas and HITL focused apps

Used for shared state, generative UI, and human-in-the-loop workflows.

Examples:

- [canvas-with-langgraph-python](https://github.com/CopilotKit/canvas-with-langgraph-python)
- [canvas-with-mastra](https://github.com/CopilotKit/canvas-with-mastra)
- [canvas-with-llamaindex](https://github.com/CopilotKit/canvas-with-llamaindex)
- [with-langgraph-fastapi-persisted-threads](https://github.com/CopilotKit/with-langgraph-fastapi-persisted-threads)

## Official Starter Ecosystem Support

The table below combines AG-UI upstream support status with CopilotKit official starter availability.

| Framework/Spec | AG-UI Upstream Status | CopilotKit Starter Availability | Starter Repositories |
|---|---|---|---|
| LangGraph | Supported | `with-*`, `canvas-*`, `coagents-*`, persisted threads | `with-langgraph-fastapi`, `with-langgraph-fastapi-persisted-threads`, `with-langgraph-js`, `with-langgraph-python`, `canvas-with-langgraph-python`, `coagents-starter-langgraph` |
| Mastra | Supported | `with-*`, `canvas-*` | `with-mastra`, `canvas-with-mastra` |
| Pydantic AI | Supported | `with-*` | `with-pydantic-ai` |
| LlamaIndex | Supported | `with-*`, `canvas-*` | `with-llamaindex`, `canvas-with-llamaindex`, `canvas-with-llamaindex-composio` |
| CrewAI Flows | Supported (CrewAI) | `with-*`, `coagents-*` | `with-crewai-flows`, `coagents-starter-crewai-flows` |
| Microsoft Agent Framework | Supported | `with-*` | `with-microsoft-agent-framework-dotnet`, `with-microsoft-agent-framework-python` |
| Google ADK | Supported | `with-*` | `with-adk` |
| AWS Strands | Supported | `with-*` | `with-strands-python` |
| Agno | Supported | `with-*` | `with-agno` |
| AG2 | Supported | demo-level only | `ag2-feature-viewer` |
| A2A Protocol | Supported | protocol starters | `with-a2a-middleware`, `with-a2a-a2ui` |
| Oracle Agent Spec | Supported | protocol/spec starter | `with-agent-spec` |
| MCP Apps | Supported | protocol/spec starter | `with-mcp-apps` |
| AWS Bedrock Agents | In Progress | none yet | (no official starter repo found) |
| OpenAI Agent SDK | In Progress | none yet | (no official starter repo found) |
| Cloudflare Agents | In Progress | none yet | (no official starter repo found) |

## Minimum Feature Comparison (Official Starters)

This comparison is based on actual `page.tsx` and `route.ts` usage in each starter.

| Starter | Runtime Adapter | Shared State | Frontend Actions | Generative UI | HITL |
|---|---|---|---|---|---|
| `with-langgraph-fastapi` | `LangGraphHttpAgent` | Yes | Yes | Yes | Yes |
| `with-langgraph-js` | `LangGraphAgent` | Yes | Yes | Yes | No |
| `with-mastra` | `MastraAgent` | Yes | Yes | Yes | Yes |
| `with-pydantic-ai` | `HttpAgent` | Yes | Yes | Yes | Yes |
| `with-llamaindex` | `LlamaIndexAgent` | Yes | Yes | Yes | No |
| `with-crewai-flows` | `HttpAgent` | Yes | Yes | Yes | No |
| `with-agno` | `HttpAgent` | No | Yes | Yes | No |
| `with-adk` | `HttpAgent` | Yes | Yes | Yes | Yes |
| `with-strands-python` | `HttpAgent` | Yes | Yes | Yes | No |
| `with-microsoft-agent-framework-dotnet` | `HttpAgent` | Yes | Yes | Yes | Yes |
| `with-microsoft-agent-framework-python` | `HttpAgent` | Yes | Yes | Yes | Yes |
| `with-langgraph-fastapi-persisted-threads` | `LangGraphHttpAgent` | Yes | Yes | Yes | Yes |

Notes:

- Most starters use Next.js `copilotRuntimeNextJSAppRouterEndpoint` as the frontend runtime bridge.
- Most non-LangGraph/Mastra starters use generic `HttpAgent` transport.
- `with-langgraph-fastapi-persisted-threads` is the explicit persisted-thread variant.

## AG-UI Supported Frameworks (Upstream)

AG-UI upstream lists first-party/partner integrations including LangGraph, CrewAI, Microsoft Agent Framework, ADK, Strands, Mastra, Pydantic AI, Agno, and LlamaIndex.

- Upstream list: [AG-UI README integrations table](https://github.com/ag-ui-protocol/ag-ui)

## AI SDK Frontend Pattern

For `@ai-sdk/react`, the common pattern is:

1. `useChat` in the browser.
2. Next.js `/api/chat` route adapts UI messages to backend request body.
3. Route passes through SSE stream and AI SDK headers.

Best in this repository when:

- frontend is mostly a chat shell
- the backend remains the only place tools execute
- you want the smallest integration surface

Related how-to:

- [Integrate AI SDK Frontend](../how-to/integrate-ai-sdk-frontend.md)

## AG-UI / CopilotKit Frontend Pattern

For CopilotKit and AG-UI, the common pattern is:

1. CopilotKit runtime runs behind a same-origin Next.js route.
2. `HttpAgent` points at Tirea's AG-UI endpoint.
3. AG-UI SSE events drive chat, shared state, activity updates, and frontend tool suspensions.
4. Frontend decisions are sent back to resume pending interactions.

Best in this repository when:

- frontend must participate in tool execution
- you need richer HITL flows
- you want persisted thread hydration plus shared state UI

Related how-to:

- [Integrate CopilotKit (AG-UI)](../how-to/integrate-copilotkit-ag-ui.md)

Related docs:

- [AI SDK Transport](https://ai-sdk.dev/docs/ai-sdk-ui/transport)
- [AI SDK `useChat`](https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat)

## Mapping to Tirea Endpoints

Run streaming:

- AG-UI: `POST /v1/ag-ui/agents/:agent_id/runs`
- AI SDK v6: `POST /v1/ai-sdk/agents/:agent_id/runs`

History:

- Raw messages: `GET /v1/threads/:id/messages`
- AG-UI encoded: `GET /v1/ag-ui/threads/:id/messages`
- AI SDK encoded: `GET /v1/ai-sdk/threads/:id/messages`

## In-Repo Integration Examples

CopilotKit:

- `examples/copilotkit-starter/README.md`
- `examples/travel-ui/README.md`
- `examples/research-ui/README.md`

AI SDK:

- `examples/ai-sdk-starter/README.md`
