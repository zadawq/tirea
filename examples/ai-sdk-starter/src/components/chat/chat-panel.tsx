import { useChatSession } from "@/hooks/use-chat-session";
import { MessageList } from "./message-list";
import { ChatInput } from "./chat-input";
import { MetricsPanel } from "./metrics-panel";

type ChatPanelProps = {
  threadId: string;
  agentId?: string;
};

export function ChatPanel({ threadId, agentId = "default" }: ChatPanelProps) {
  const {
    messages,
    sendMessage,
    status,
    error,
    addToolApprovalResponse,
    addToolOutput,
    historyLoaded,
    metrics,
    askAnswers,
    setAskAnswers,
  } = useChatSession(threadId, agentId);

  const isLoading = status === "streaming" || status === "submitted";

  if (!historyLoaded) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-400">
        Loading...
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <MessageList
        messages={messages}
        isLoading={isLoading}
        askAnswers={askAnswers}
        onAskAnswerChange={(toolCallId, value) =>
          setAskAnswers((prev) => ({ ...prev, [toolCallId]: value }))
        }
        onApprove={(id) => addToolApprovalResponse({ id, approved: true })}
        onDeny={(id) => addToolApprovalResponse({ id, approved: false })}
        onAskSubmit={async (toolCallId, answer) => {
          await addToolOutput({
            tool: "askUserQuestion" as never,
            toolCallId,
            state: "output-available",
            output: { message: answer } as never,
          });
          setAskAnswers((prev) => ({ ...prev, [toolCallId]: "" }));
        }}
      />
      {error && (
        <div className="bg-red-50 px-4 py-2 text-sm text-red-600">
          Error: {error.message}
        </div>
      )}
      <MetricsPanel metrics={metrics} />
      <ChatInput
        onSend={(text) => sendMessage({ text })}
        disabled={isLoading}
      />
    </div>
  );
}
