import type { CSSProperties, ReactNode } from "react";

type SplitScreenProps = {
  rootTestId: string;
  pageStyle: CSSProperties;
  chatHint: string;
  left: ReactNode;
  chat: ReactNode;
  chatTheme?: "light" | "dark";
};

export function SplitScreen({
  rootTestId,
  pageStyle,
  chatHint,
  left,
  chat,
  chatTheme = "light",
}: SplitScreenProps) {
  const isDark = chatTheme === "dark";
  return (
    <main
      data-testid={rootTestId}
      className="relative isolate min-h-screen overflow-hidden text-white"
      style={pageStyle}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(500px_300px_at_5%_5%,rgba(255,255,255,0.3),transparent_70%),radial-gradient(600px_400px_at_95%_15%,rgba(236,252,203,0.22),transparent_72%)]"
      />
      <div className="grid h-screen grid-cols-1 lg:grid-cols-[minmax(0,1fr)_460px]">
        <section className="min-h-0 min-w-0 overflow-hidden">
          <div className="h-full overflow-y-auto px-4 py-5 lg:px-5">
            {left}
          </div>
        </section>
        <aside className="min-h-[66vh] border-t border-white/30 bg-white/10 p-2 backdrop-blur-md lg:min-h-0 lg:border-l lg:border-t-0">
          <div
            className={
              isDark
                ? "flex h-full flex-col overflow-hidden rounded-2xl border border-slate-700 bg-slate-900/95 shadow-[0_18px_36px_rgba(2,6,23,0.5)]"
                : "flex h-full flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white/95 shadow-[0_18px_36px_rgba(15,23,42,0.16)]"
            }
          >
            <header
              className={
                isDark
                  ? "border-b border-slate-700 bg-gradient-to-b from-slate-900 to-slate-800 px-4 py-3"
                  : "border-b border-slate-200 bg-gradient-to-b from-slate-50 to-slate-100 px-4 py-3"
              }
            >
              <h2 className={isDark ? "m-0 text-base font-semibold text-slate-100" : "m-0 text-base font-semibold text-slate-900"}>
                tirea assistant
              </h2>
              <p className={isDark ? "mt-1 text-xs text-slate-300" : "mt-1 text-xs text-slate-600"}>
                {chatHint}
              </p>
            </header>
            <div className="min-h-0 flex-1 overflow-hidden">{chat}</div>
          </div>
        </aside>
      </div>
    </main>
  );
}
