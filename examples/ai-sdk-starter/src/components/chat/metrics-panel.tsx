import type { InferenceMetrics } from "@/hooks/use-chat-session";

export function MetricsPanel({ metrics }: { metrics: InferenceMetrics[] }) {
  if (metrics.length === 0) return null;

  return (
    <div
      data-testid="metrics-panel"
      className="border-t border-slate-200 bg-blue-50/80 px-4 py-2 text-xs"
    >
      <strong className="text-slate-700">Token Usage</strong>
      {metrics.map((m, i) => {
        const totalTokens =
          m.usage?.total_tokens ??
          (m.usage?.prompt_tokens ?? 0) + (m.usage?.completion_tokens ?? 0);
        return (
          <div key={i} data-testid="metrics-entry" className="mt-0.5 text-slate-600">
            {m.model}: {totalTokens > 0 ? `${totalTokens} tokens` : "no usage data"}{" "}
            ({m.duration_ms}ms)
          </div>
        );
      })}
    </div>
  );
}
