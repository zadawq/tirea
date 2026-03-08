# Integrate AI SDK Frontend

Use this when your web app uses `@ai-sdk/react` and backend is `tirea-agentos-server`.

Choose this path when you want the simplest chat-style integration and your frontend does not need AG-UI-specific shared state or frontend tool orchestration.

## Best Fit

AI SDK is usually the right choice when:

- your UI is primarily chat-first
- you want `useChat` with minimal frontend runtime glue
- tools mostly execute on the backend
- you want to hydrate thread history into an AI SDK-compatible message stream

Choose AG-UI/CopilotKit instead when you need richer shared state, frontend-executed tools, or canvas-style interactions.

## Prerequisites

- Backend is reachable (default `http://localhost:8080`).
- AI SDK routes are enabled:
  - `POST /v1/ai-sdk/agents/:agent_id/runs`
  - `GET /v1/ai-sdk/agents/:agent_id/runs/:chat_id/stream`
  - `GET /v1/ai-sdk/threads/:id/messages`

## Minimum Architecture

```text
Browser (useChat)
  -> Next.js route (/api/chat)
  -> Tirea AI SDK endpoint (/v1/ai-sdk/agents/:agent_id/runs)
  -> SSE stream back to browser
```

Thread history hydration:

```text
Browser
  -> GET /v1/ai-sdk/threads/:id/messages
  -> backend returns AI SDK-encoded message history
```

## Steps

1. Install frontend deps.

```bash
npm install @ai-sdk/react ai
```

2. Create `app/api/chat/route.ts` as pass-through proxy.

```ts,ignore
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8080";
const AGENT_ID = process.env.AGENT_ID ?? "default";

export async function POST(req: Request) {
  const incoming = await req.json();
  const agentId = incoming.agentId ?? AGENT_ID;

  const sessionId =
    req.headers.get("x-session-id") ??
    incoming.id ??
    `ai-sdk-${crypto.randomUUID()}`;

  const upstream = await fetch(
    `${BACKEND_URL}/v1/ai-sdk/agents/${agentId}/runs`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...incoming, id: sessionId }),
    },
  );

  if (!upstream.ok) {
    return new Response(await upstream.text(), { status: upstream.status });
  }
  if (!upstream.body) {
    return new Response("upstream body missing", { status: 502 });
  }

  return new Response(upstream.body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-vercel-ai-ui-message-stream": "v1",
      "x-session-id": sessionId,
    },
  });
}
```

3. Build transport with explicit `id/messages` payload.

```tsx,ignore
"use client";

import { useMemo } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";

export default function ChatPage() {
  const sessionId = "ai-sdk-demo-session";

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        headers: { "x-session-id": sessionId },
        prepareSendMessagesRequest: ({ messages, trigger, messageId }) => {
          const lastAssistantIndex = (() => {
            for (let i = messages.length - 1; i >= 0; i -= 1) {
              if (messages[i]?.role === "assistant") return i;
            }
            return -1;
          })();

          const newUserMessages = messages
            .slice(lastAssistantIndex + 1)
            .filter((m) => m.role === "user");

          return {
            body: {
              id: sessionId,
              runId: crypto.randomUUID(),
              messages: trigger === "regenerate-message" ? [] : newUserMessages,
              ...(trigger ? { trigger } : {}),
              ...(messageId ? { messageId } : {}),
            },
          };
        },
      }),
    [sessionId],
  );

  const { messages, sendMessage, status } = useChat({ transport });
  // render + sendMessage({ text })
}
```

4. Optional: load history from `GET /v1/ai-sdk/threads/:id/messages` before first render.

## Backend / Frontend Boundary

In the AI SDK integration path:

- backend tools execute in `tirea-agentos-server`
- frontend mainly owns chat rendering and local UI state
- thread state remains backend-owned
- the Next.js route is usually only a transport adapter, not the source of truth

A practical rule:

- durable conversation and tool state live in Tirea
- browser-only visual state stays in the frontend
- if you need frontend-executed tools with suspend/resume semantics, AG-UI is usually a better fit

## Request Shape Mapping

The frontend sends AI SDK UI messages, but Tirea expects the v6 HTTP shape:

- `id` maps to thread id
- `messages` contains the new user messages to submit
- `runId` is used for decision-forwarding correlation
- `trigger` and `messageId` are required for regenerate flows

The route adapter is responsible for preserving this shape. Do not send legacy `sessionId/input` bodies.

## Resume And Approval Flow

AI SDK integration supports suspension decision forwarding, but the transport is still chat-centric.

Typical flow:

1. backend suspends a tool call and emits AI SDK stream events
2. frontend renders an approval UI from the streamed message parts
3. frontend submits a decision-only payload using the same `id` and `runId`
4. server forwards the decision to the active run when possible
5. stream resumes or a new execution path is started if no active run is found

This path works well for approval dialogs, but it is less expressive than AG-UI for frontend tool execution.

## Verify

- `/api/chat` responds with `text/event-stream`.
- Response includes `x-vercel-ai-ui-message-stream: v1`.
- Normal chat streams tokens.
- Regenerate flow (`trigger=regenerate-message`) works with `messageId`.
- History hydration from `GET /v1/ai-sdk/threads/:id/messages` replays the same thread in the browser.

## Common Errors

- Sending legacy `sessionId/input` payload to backend HTTP v6 route.
- Missing `id` (thread id) in forwarded payload.
- Missing stream headers, causing client parser failures.

## Related Example

- `examples/ai-sdk-starter/README.md` is the canonical AI SDK v6 integration in this repo

## Key Files

- `examples/ai-sdk-starter/src/lib/transport.ts`
- `examples/ai-sdk-starter/src/pages/playground-page.tsx`
- `examples/ai-sdk-starter/src/lib/api-client.ts`

## Related

- [AI SDK v6 Protocol](../reference/protocols/ai-sdk-v6.md)
- [Ecosystem Integrations](../reference/ecosystem-integrations.md)
- [HTTP API](../reference/http-api.md)
- `examples/ai-sdk-starter/src/lib/transport.ts`
