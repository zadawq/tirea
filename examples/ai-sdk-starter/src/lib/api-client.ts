import type { UIMessage } from "@ai-sdk/react";

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL ?? "http://localhost:38080";

export function chatApiUrl(agentId: string): string {
  return `${BACKEND_URL}/v1/ai-sdk/agents/${agentId}/runs`;
}

export function historyApiUrl(sessionId: string): string {
  return `${BACKEND_URL}/v1/ai-sdk/threads/${encodeURIComponent(sessionId)}/messages?limit=200`;
}

export function threadsApiUrl(): string {
  return `${BACKEND_URL}/v1/threads?offset=0&limit=200`;
}

export async function fetchHistory(sessionId: string): Promise<UIMessage[]> {
  try {
    const res = await fetch(historyApiUrl(sessionId));
    if (!res.ok) return [];
    const data = await res.json();
    return data.messages ?? [];
  } catch {
    return [];
  }
}

export async function fetchThreadIds(): Promise<string[]> {
  try {
    const res = await fetch(threadsApiUrl());
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.items) ? data.items : [];
  } catch {
    return [];
  }
}
