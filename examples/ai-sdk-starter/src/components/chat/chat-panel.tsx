import { useChatSession } from "@/hooks/use-chat-session";
import { MessageList } from "./message-list";
import { ChatInput } from "./chat-input";
import { MetricsPanel } from "./metrics-panel";
import { ToolProgressPanel } from "./tool-progress-panel";

type ChatPanelProps = {
  threadId: string;
  agentId?: string;
  themeMode?: "light" | "dark";
};

export function ChatPanel({
  threadId,
  agentId = "default",
  themeMode = "light",
}: ChatPanelProps) {
  const {
    messages,
    sendMessage,
    status,
    error,
    addToolApprovalResponse,
    addToolOutput,
    historyLoaded,
    metrics,
    toolProgress,
    askAnswers,
    setAskAnswers,
  } = useChatSession(threadId, agentId);

  const isLoading = status === "streaming" || status === "submitted";

  if (!historyLoaded) {
    const loadingClass =
      themeMode === "dark" ? "text-slate-400" : "text-slate-400";
    return (
      <div className={`flex h-full items-center justify-center text-sm ${loadingClass}`}>
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
        themeMode={themeMode}
      />
      {error && (
        <div
          className={
            themeMode === "dark"
              ? "bg-red-950/30 px-4 py-2 text-sm text-red-300"
              : "bg-red-50 px-4 py-2 text-sm text-red-600"
          }
        >
          Error: {error.message}
        </div>
      )}
      <ToolProgressPanel progressByNodeId={toolProgress} />
      <MetricsPanel metrics={metrics} />
      <ChatInput
        onSend={(text) => sendMessage({ text })}
        disabled={isLoading}
        themeMode={themeMode}
      />
    </div>
  );
}
