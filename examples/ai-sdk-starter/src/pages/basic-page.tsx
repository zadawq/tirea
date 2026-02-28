import { type CSSProperties, useState } from "react";
import { SplitScreen } from "@/components/layout/split-screen";
import { NavTabs } from "@/components/layout/nav-tabs";
import { ChatPanel } from "@/components/chat/chat-panel";
import { SharedStatePanel } from "@/components/demo/shared-state-panel";
import { RecommendedActions } from "@/components/demo/recommended-actions";
import { RECOMMENDED_ACTIONS } from "@/lib/recommended-actions";
import { getSessionId } from "@/lib/session";

const panelCardClass = "rounded-xl border border-slate-200 bg-white/75 p-3";
const sectionLabelClass =
  "mb-2 inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-bold text-blue-700";
const subtleHintClass = "mt-2 text-sm text-slate-600";

export function BasicPage() {
  const [sessionId] = useState(getSessionId);
  const [todos, setTodos] = useState<string[]>([
    "Review tirea integration",
    "Run smoke test",
  ]);

  const pageStyle = {
    backgroundColor: "#3b82f6",
  } as CSSProperties;

  const leftPane = (
    <div className="mx-auto max-w-[1040px]">
      <div className="inline-flex items-center gap-2 rounded-full border border-white/50 bg-slate-900/30 px-3 py-1 text-xs font-semibold tracking-wide text-slate-100">
        AI SDK x tirea starter
      </div>
      <h1
        data-testid="page-title"
        className="mt-3 text-4xl font-bold tracking-tight text-white md:text-5xl"
      >
        with-tirea basic
      </h1>
      <p className="mt-2 max-w-[760px] text-sm text-slate-100/95 md:text-base">
        AI SDK basic demo with backend tools, tool approval, and interactive dialogs.
      </p>
      <NavTabs />
      <RecommendedActions title="Recommended Actions" actions={RECOMMENDED_ACTIONS} />

      <div className="mt-4 grid gap-3 rounded-2xl border border-white/60 bg-white/80 p-5 text-slate-900 shadow-[0_20px_45px_rgba(15,23,42,0.14)] backdrop-blur">
        <SharedStatePanel
          label="Local State"
          title="1) Local State Management"
          prompt="Manage todos locally and interact with backend tools."
          todos={todos}
          defaultInput="Write a Playwright smoke test"
          onAddTodo={(text) => setTodos((prev) => [...prev, text])}
          onClearTodos={() => setTodos([])}
        />

        <section className={panelCardClass}>
          <div className={sectionLabelClass}>Backend Tools</div>
          <h3 className="text-lg font-semibold text-slate-900">
            2) Backend Tools (weather, stock, notes)
          </h3>
          <p data-testid="backend-tools-prompt" className={subtleHintClass}>
            Prompt: &quot;What&apos;s the weather in San Francisco?&quot; or &quot;Show AAPL stock
            price.&quot;
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Backend tools execute on the Rust agent and return structured data displayed as
            cards in chat.
          </p>
        </section>

        <section className={panelCardClass}>
          <div className={sectionLabelClass}>Tool Approval</div>
          <h3 className="text-lg font-semibold text-slate-900">
            3) Tool Approval (PermissionConfirm)
          </h3>
          <p data-testid="approval-prompt" className={subtleHintClass}>
            Use an agent with the permission plugin to see approval dialogs for tool
            execution.
          </p>
        </section>

        <section className={panelCardClass}>
          <div className={sectionLabelClass}>Interactive Dialog</div>
          <h3 className="text-lg font-semibold text-slate-900">
            4) Interactive Dialog (askUserQuestion)
          </h3>
          <p data-testid="dialog-prompt" className={subtleHintClass}>
            The agent can ask follow-up questions. Your answers are sent back as tool
            outputs.
          </p>
        </section>
      </div>
    </div>
  );

  return (
    <SplitScreen
      rootTestId="basic-root"
      pageStyle={pageStyle}
      chatHint="Try: ask for weather, stock prices, or notes."
      left={leftPane}
      chat={<ChatPanel threadId={sessionId} />}
    />
  );
}
