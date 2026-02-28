export type StarterAction = {
  id: string;
  title: string;
  capability: string;
  prompt: string;
};

export const RECOMMENDED_ACTIONS: StarterAction[] = [
  {
    id: "weather",
    title: "Weather Check",
    capability: "Backend Tool",
    prompt: "What's the weather in San Francisco?",
  },
  {
    id: "stock",
    title: "Stock Quote",
    capability: "Backend Tool",
    prompt: "Show me the latest AAPL stock price.",
  },
  {
    id: "note",
    title: "Append Note",
    capability: "Stateful Tool",
    prompt: "Append a note: verify AG-UI event ordering.",
  },
  {
    id: "multi-tool",
    title: "Multi-Tool",
    capability: "Parallel Tools",
    prompt: "Get weather in Tokyo and stock price for NVDA.",
  },
];
