import { useChat, type UIMessage } from "@ai-sdk/react";
import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { fetchHistory } from "@/lib/api-client";
import { createTransport } from "@/lib/transport";

export interface InferenceMetrics {
  model: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  duration_ms: number;
}

export function useChatSession(threadId: string, agentId = "default") {
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | undefined>(undefined);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [metrics, setMetrics] = useState<InferenceMetrics[]>([]);
  const [askAnswers, setAskAnswers] = useState<Record<string, string>>({});
  const autoSubmittedIds = useRef<Set<string>>(new Set());
  const historyFetched = useRef(false);

  useEffect(() => {
    if (!threadId || historyFetched.current) return;
    historyFetched.current = true;

    fetchHistory(threadId).then((messages) => {
      setInitialMessages(messages.length > 0 ? messages : undefined);
      setHistoryLoaded(true);
    });
  }, [threadId]);

  const transport = useMemo(
    () => createTransport(threadId, agentId),
    [threadId, agentId],
  );

  const onData = useCallback((dataPart: { type: string; data: unknown }) => {
    if (dataPart.type === "data-inference-complete") {
      setMetrics((prev) => [...prev, dataPart.data as InferenceMetrics]);
    }
  }, []);

  const sendAutomaticallyWhen = useCallback(
    ({ messages }: { messages: UIMessage[] }) => {
      const last = messages[messages.length - 1];
      if (!last || last.role !== "assistant") return false;

      const interactionIds = last.parts.flatMap((part) => {
        if (
          (part.type === "dynamic-tool" || part.type.startsWith("tool-")) &&
          "state" in part &&
          typeof part.state === "string"
        ) {
          const state = part.state;
          const toolCallId =
            "toolCallId" in part && typeof part.toolCallId === "string"
              ? part.toolCallId
              : undefined;
          const toolName =
            "toolName" in part && typeof part.toolName === "string"
              ? part.toolName
              : part.type.startsWith("tool-")
                ? part.type.slice("tool-".length)
                : undefined;

          if (state === "approval-responded") {
            const approval = (part as { approval?: { id?: unknown } }).approval;
            if (approval && typeof approval.id === "string") return [approval.id];
            if (toolCallId) return [toolCallId];
          }

          if (
            toolName === "askUserQuestion" &&
            (state === "output-available" || state === "output-denied" || state === "output-error")
          ) {
            if (toolCallId) return [toolCallId];
          }
        }
        return [];
      });

      let shouldSend = false;
      for (const id of interactionIds) {
        if (!autoSubmittedIds.current.has(id)) {
          autoSubmittedIds.current.add(id);
          shouldSend = true;
        }
      }
      return shouldSend;
    },
    [],
  );

  const chat = useChat({
    messages: initialMessages,
    transport,
    onData: onData as never,
    sendAutomaticallyWhen,
  });

  return {
    ...chat,
    historyLoaded,
    metrics,
    askAnswers,
    setAskAnswers,
  };
}
