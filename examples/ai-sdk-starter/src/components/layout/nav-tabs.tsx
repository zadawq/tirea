import { Link, useLocation } from "react-router";

type StarterMode = "canvas" | "basic" | "threads";

const TABS: { mode: StarterMode; href: string; label: string }[] = [
  { mode: "canvas", href: "/", label: "Canvas" },
  { mode: "basic", href: "/basic", label: "Basic" },
  { mode: "threads", href: "/threads", label: "Threads" },
];

export function NavTabs() {
  const { pathname } = useLocation();

  const activeMode: StarterMode =
    pathname === "/basic"
      ? "basic"
      : pathname === "/threads"
        ? "threads"
        : "canvas";

  const tabClass = (active: boolean) =>
    active
      ? "rounded-full border border-cyan-700 bg-cyan-700 px-3 py-1.5 text-sm font-semibold text-cyan-50 shadow-[0_10px_24px_rgba(14,116,144,0.24)]"
      : "rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm font-semibold text-slate-800 transition hover:-translate-y-px hover:border-cyan-600 hover:bg-cyan-50 hover:shadow-[0_8px_20px_rgba(14,116,144,0.16)]";

  return (
    <nav
      className="mt-3 flex w-fit flex-wrap items-center gap-2 rounded-full border border-slate-200 bg-white/80 p-1 backdrop-blur"
      data-testid="starter-mode-tabs"
    >
      {TABS.map((tab) => (
        <Link
          key={tab.mode}
          to={tab.href}
          data-testid={`${tab.mode}-link`}
          className={tabClass(activeMode === tab.mode)}
        >
          {tab.label}
        </Link>
      ))}
    </nav>
  );
}
