import { type CSSProperties, useState } from "react";
import { SplitScreen } from "@/components/layout/split-screen";
import { NavTabs } from "@/components/layout/nav-tabs";
import { ChatPanel } from "@/components/chat/chat-panel";
import { SharedStatePanel } from "@/components/demo/shared-state-panel";
import { RecommendedActions } from "@/components/demo/recommended-actions";
import { TodoBoard, type Todo, createTodo } from "@/components/demo/todo-board";
import { RECOMMENDED_ACTIONS } from "@/lib/recommended-actions";
import { getSessionId } from "@/lib/session";

export function CanvasPage() {
  const [sessionId] = useState(getSessionId);
  const [themeMode, setThemeMode] = useState<"light" | "dark">("light");
  const [todos, setTodos] = useState<Todo[]>([]);

  const isDark = themeMode === "dark";
  const surface = isDark ? "#0f172a" : "#f8fafc";
  const text = isDark ? "#e2e8f0" : "#0f172a";
  const pageStyle = { backgroundColor: surface, color: text } as CSSProperties;

  const sharedTodos = todos.map(
    (t) => `${t.status === "completed" ? "Done" : "Todo"}: ${t.title}`,
  );

  const subtitleClass = isDark ? "text-slate-200" : "text-slate-700";
  const boardCardClass = isDark
    ? "rounded-xl border border-slate-600 bg-slate-900/70 p-3"
    : "rounded-xl border border-slate-200 bg-white/75 p-3";

  const leftPane = (
    <div className="mx-auto max-w-[1040px]">
      <div className="inline-flex items-center gap-2 rounded-full border border-white/50 bg-slate-900/30 px-3 py-1 text-xs font-semibold tracking-wide text-slate-100">
        AI SDK x tirea starter
      </div>
      <h1
        data-testid="page-title"
        className="mt-3 text-4xl font-bold tracking-tight text-white md:text-5xl"
      >
        with-tirea canvas
      </h1>
      <p className={`mt-2 max-w-[760px] text-sm md:text-base ${subtitleClass}`}>
        AI SDK canvas demo with todo board, backend tools, and split-screen chat.
      </p>
      <NavTabs />
      <RecommendedActions title="Recommended Actions" actions={RECOMMENDED_ACTIONS} />

      <section className="mt-4 grid gap-3 rounded-2xl border border-white/60 bg-white/80 p-5 text-slate-900 shadow-[0_20px_45px_rgba(15,23,42,0.14)] backdrop-blur">
        <div className={boardCardClass}>
          <div className="mb-2 inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-bold text-blue-700">
            Canvas Controls
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              data-testid="toggle-theme"
              onClick={() => setThemeMode((p) => (p === "dark" ? "light" : "dark"))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-800 transition hover:-translate-y-px hover:border-slate-400 hover:bg-slate-50"
            >
              Toggle Theme ({themeMode})
            </button>
          </div>
        </div>

        <SharedStatePanel
          label="Shared State"
          title="Shared State (derived from tools)"
          prompt="Append a note: verify AG-UI event ordering."
          todos={sharedTodos}
          defaultInput="Task from shared-state panel"
          onAddTodo={(text) => setTodos([...todos, createTodo(todos.length, text)])}
          onClearTodos={() => setTodos([])}
        />

        <div className={boardCardClass}>
          <div className="mb-2 inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-bold text-blue-700">
            Todo Board
          </div>
          <TodoBoard todos={todos} onChange={setTodos} />
        </div>
      </section>
    </div>
  );

  return (
    <SplitScreen
      rootTestId="canvas-root"
      pageStyle={pageStyle}
      chatHint="Try: get weather, stock price, or append a note."
      left={leftPane}
      chat={<ChatPanel threadId={sessionId} />}
    />
  );
}
