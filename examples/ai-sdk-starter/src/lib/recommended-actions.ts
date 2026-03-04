export type StarterAction = {
  id: string;
  title: string;
  capability: string;
  prompt: string;
  agentId?: "default" | "permission" | "stopper";
};

export const RECOMMENDED_ACTIONS: StarterAction[] = [
  {
    id: "demo-weather",
    title: "Weather Check",
    capability: "Backend Tool",
    prompt: "RUN_WEATHER_TOOL",
    agentId: "default",
  },
  {
    id: "demo-stock",
    title: "Stock Quote",
    capability: "Backend Tool",
    prompt: "RUN_STOCK_TOOL",
    agentId: "default",
  },
  {
    id: "demo-note",
    title: "Append Note",
    capability: "Stateful Tool",
    prompt: "RUN_APPEND_NOTE Please append a note saying: starter-demo-note",
    agentId: "default",
  },
  {
    id: "demo-server-info",
    title: "Server Info",
    capability: "Backend Tool",
    prompt: "RUN_SERVER_INFO",
    agentId: "default",
  },
  {
    id: "demo-progress",
    title: "Progress Demo",
    capability: "Tool Progress",
    prompt: "RUN_PROGRESS_DEMO",
    agentId: "default",
  },
  {
    id: "demo-failing",
    title: "Failure Path",
    capability: "Backend Error",
    prompt: "RUN_FAILING_TOOL",
    agentId: "default",
  },
  {
    id: "demo-ask-user",
    title: "Ask User",
    capability: "Frontend Tool",
    prompt: "RUN_ASK_USER_TOOL",
    agentId: "default",
  },
  {
    id: "demo-bg-color",
    title: "Background Color",
    capability: "Client-side Action",
    prompt: "RUN_BG_TOOL",
    agentId: "default",
  },
  {
    id: "demo-permission-allow",
    title: "Permission Approve",
    capability: "Approval Flow",
    prompt: "RUN_SERVER_INFO",
    agentId: "permission",
  },
  {
    id: "demo-permission-deny",
    title: "Permission Deny",
    capability: "Approval Flow",
    prompt: "RUN_SERVER_INFO",
    agentId: "permission",
  },
  {
    id: "demo-finish",
    title: "Stop On Tool",
    capability: "Stop Policy",
    prompt: "RUN_FINISH_TOOL",
    agentId: "stopper",
  },
];
