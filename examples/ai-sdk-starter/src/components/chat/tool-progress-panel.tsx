import type { ToolCallProgressNode } from "@/hooks/use-chat-session";

type ToolProgressPanelProps = {
  progressByNodeId: Record<string, ToolCallProgressNode>;
};

function formatPercent(progress?: number) {
  if (progress === undefined) return "—";
  return `${Math.round(progress * 100)}%`;
}

export function ToolProgressPanel({ progressByNodeId }: ToolProgressPanelProps) {
  const entries = Object.values(progressByNodeId).sort(
    (a, b) => (b.updated_at_ms ?? 0) - (a.updated_at_ms ?? 0),
  );
  if (entries.length === 0) return null;

  return (
    <div
      data-testid="tool-progress-panel"
      className="border-t border-slate-200 bg-emerald-50/60 px-4 py-2 text-xs"
    >
      <strong className="text-slate-700">Tool Call Progress</strong>
      <div className="mt-1 space-y-1">
        {entries.slice(0, 6).map((node) => (
          <div key={node.node_id} className="text-slate-600">
            <span className="font-medium">{node.tool_name ?? node.call_id ?? node.node_id}</span>
            <span className="ml-2 uppercase">{node.status}</span>
            <span className="ml-2">{formatPercent(node.progress)}</span>
            {node.message ? <span className="ml-2">{node.message}</span> : null}
          </div>
        ))}
      </div>
    </div>
  );
}
