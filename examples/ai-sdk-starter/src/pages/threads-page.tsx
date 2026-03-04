import { type CSSProperties } from "react";
import { SplitScreen } from "@/components/layout/split-screen";
import { NavTabs } from "@/components/layout/nav-tabs";
import { ChatPanel } from "@/components/chat/chat-panel";
import { RecommendedActions } from "@/components/demo/recommended-actions";
import { RECOMMENDED_ACTIONS } from "@/lib/recommended-actions";
import { useThreads } from "@/hooks/use-threads";

const pageStyle = {
  backgroundColor: "#e2e8f0",
} as CSSProperties;

export function ThreadsPage() {
  const { threads, activeThreadId, setActiveThreadId, createThread, loaded } =
    useThreads();

  const leftPane = (
    <div className="mx-auto max-w-[1040px]">
      <div className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold tracking-wide text-slate-700">
        AI SDK x tirea starter
      </div>
      <h1
        data-testid="page-title"
        className="mt-3 text-4xl font-bold tracking-tight text-slate-900 md:text-5xl"
      >
        with-tirea threads
      </h1>
      <p className="mt-2 max-w-[760px] text-sm text-slate-700 md:text-base">
        Backend-persisted thread demo: switch threads, keep history and state.
      </p>
      <NavTabs />
      <RecommendedActions title="Recommended Actions" actions={RECOMMENDED_ACTIONS} />

      <div className="mt-4 grid gap-3 rounded-2xl border border-white/60 bg-white/80 p-5 text-slate-900 shadow-[0_20px_45px_rgba(15,23,42,0.14)] backdrop-blur">
        <section className="rounded-xl border border-slate-200 bg-white/75 p-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="mb-2 inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-bold text-blue-700">
                Persisted Threads
              </div>
              <h2 className="text-xl font-semibold text-slate-900">
                Thread State
              </h2>
              <p
                data-testid="active-thread"
                className="mt-1 text-sm text-slate-700"
              >
                Active thread:{" "}
                {activeThreadId ? activeThreadId.slice(0, 16) + "..." : "none"}
              </p>
            </div>
            <button
              data-testid="thread-create"
              onClick={createThread}
              className="min-w-32 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-800 transition hover:-translate-y-px hover:border-slate-400 hover:bg-slate-50"
            >
              + New Thread
            </button>
          </div>
          <p
            data-testid="thread-prompt"
            className="mt-2 text-sm text-slate-600"
          >
            Create or switch threads. Each thread has its own chat history
            persisted on the backend.
          </p>
          <div
            data-testid="thread-list"
            className="mt-3 grid grid-cols-[repeat(auto-fill,minmax(170px,1fr))] gap-2"
          >
            {threads.map((thread) => (
              <button
                key={thread.id}
                data-testid={`thread-item-${thread.id}`}
                onClick={() => setActiveThreadId(thread.id)}
                className={`rounded-lg border px-3 py-2 text-left transition ${
                  activeThreadId === thread.id
                    ? "border-cyan-700 bg-cyan-50 text-cyan-900"
                    : "border-slate-200 bg-white text-slate-900 hover:-translate-y-px hover:border-slate-300"
                }`}
              >
                <div className="text-sm font-semibold">{thread.title}</div>
                <div className="text-xs text-slate-500">
                  {thread.id.slice(0, 12)}
                </div>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );

  if (!loaded) {
    return (
      <SplitScreen
        rootTestId="threads-root"
        pageStyle={pageStyle}
        chatHint="Loading threads..."
        left={leftPane}
        chat={
          <div className="flex h-full items-center justify-center text-sm text-slate-400">
            Loading...
          </div>
        }
      />
    );
  }

  return (
    <SplitScreen
      rootTestId="threads-root"
      pageStyle={pageStyle}
      chatHint="Try switching threads and chatting."
      chatTheme="light"
      rootTextClass="text-slate-900"
      left={leftPane}
      chat={
        activeThreadId ? (
          <ChatPanel key={activeThreadId} threadId={activeThreadId} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">
            Select or create a thread to start chatting.
          </div>
        )
      }
    />
  );
}
