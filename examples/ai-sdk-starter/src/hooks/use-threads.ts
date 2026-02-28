import { useState, useEffect } from "react";
import { fetchThreadIds } from "@/lib/api-client";

export type Thread = {
  id: string;
  title: string;
};

export function useThreads() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string>("");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;

    fetchThreadIds().then((ids) => {
      if (cancelled) return;
      if (ids.length > 0) {
        const loaded = ids.map((id, i) => ({ id, title: `Thread ${i + 1}` }));
        setThreads(loaded);
        setActiveThreadId(loaded[0].id);
      }
      setLoaded(true);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const createThread = () => {
    const id = crypto.randomUUID();
    const thread = { id, title: `Thread ${threads.length + 1}` };
    setThreads((prev) => [...prev, thread]);
    setActiveThreadId(id);
  };

  return {
    threads,
    activeThreadId,
    setActiveThreadId,
    createThread,
    loaded,
  };
}
