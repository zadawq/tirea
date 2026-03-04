export type PlaygroundMode = "conversation" | "tools" | "canvas";

const PLAYGROUND_TABS: { mode: PlaygroundMode; label: string }[] = [
  { mode: "conversation", label: "Conversation" },
  { mode: "tools", label: "Tools" },
  { mode: "canvas", label: "Canvas" },
];

type NavTabsProps = {
  mode: PlaygroundMode;
  onModeChange: (mode: PlaygroundMode) => void;
};

export function NavTabs({ mode, onModeChange }: NavTabsProps) {

  const tabClass = (active: boolean) =>
    active
      ? "rounded-full border border-cyan-700 bg-cyan-700 px-3 py-1.5 text-sm font-semibold text-cyan-50 shadow-[0_10px_24px_rgba(14,116,144,0.24)]"
      : "rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm font-semibold text-slate-800 transition hover:-translate-y-px hover:border-cyan-600 hover:bg-cyan-50 hover:shadow-[0_8px_20px_rgba(14,116,144,0.16)]";

  return (
    <nav
      className="mt-3 flex w-fit flex-wrap items-center gap-2 rounded-full border border-slate-200 bg-white/80 p-1 backdrop-blur"
      data-testid="starter-mode-tabs"
    >
      {PLAYGROUND_TABS.map((tab) => (
        <button
          key={tab.mode}
          type="button"
          data-testid={`${tab.mode}-link`}
          className={tabClass(mode === tab.mode)}
          onClick={() => onModeChange(tab.mode)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
