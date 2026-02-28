import { DefaultChatTransport } from "ai";
import { chatApiUrl } from "./api-client";

export function createTransport(
  sessionId: string,
  agentId: string,
): DefaultChatTransport {
  return new DefaultChatTransport({
    api: chatApiUrl(agentId),
    headers: { "x-session-id": sessionId },
    prepareSendMessagesRequest: ({ messages, trigger, messageId }) => {
      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
      return {
        body: {
          id: sessionId,
          runId: crypto.randomUUID(),
          messages:
            trigger === "regenerate-message"
              ? []
              : lastUserMsg
                ? [lastUserMsg]
                : [],
          ...(trigger ? { trigger } : {}),
          ...(messageId ? { messageId } : {}),
        },
      };
    },
  });
}
