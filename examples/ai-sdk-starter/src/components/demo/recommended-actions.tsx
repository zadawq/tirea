import { useState } from "react";
import type { StarterAction } from "@/lib/recommended-actions";

export function RecommendedActions({
  title,
  actions,
}: {
  title: string;
  actions: StarterAction[];
}) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copyPrompt = async (id: string, prompt: string) => {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopiedId(id);
      setTimeout(
        () => setCopiedId((current) => (current === id ? null : current)),
        1200,
      );
    } catch {
      setCopiedId(null);
    }
  };

  return (
    <section
      className="mt-4 rounded-2xl border border-white/60 bg-white/80 p-4 text-slate-900 shadow-[0_20px_45px_rgba(15,23,42,0.14)] backdrop-blur"
      data-testid="recommended-actions"
    >
      <h2 className="m-0 text-slate-900">{title}</h2>
      <p className="mt-2 text-sm text-slate-600">
        Click to copy prompt, then paste into the chat.
      </p>
      <div className="mt-3 grid gap-2">
        {actions.map((action) => (
          <div
            key={action.id}
            className="rounded-xl border border-slate-200 bg-white/80 px-3 py-2"
            data-testid={`recommended-action-${action.id}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                    ai-sdk
                  </span>
                  <span className="text-xs text-cyan-800">
                    {action.capability}
                  </span>
                </div>
                <p className="mt-1 text-sm font-semibold text-slate-900">
                  {action.title}
                </p>
                <p className="mt-1 text-sm text-slate-700">
                  &quot;{action.prompt}&quot;
                </p>
              </div>
              <button
                type="button"
                className="shrink-0 rounded-lg border border-slate-300 bg-slate-50 px-2 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100"
                onClick={() => copyPrompt(action.id, action.prompt)}
              >
                {copiedId === action.id ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
