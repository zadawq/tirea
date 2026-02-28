const STORAGE_KEY = "tirea-session-id";

export function getSessionId(): string {
  let id = localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = `ai-sdk-${crypto.randomUUID()}`;
    localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}

export function setSessionId(id: string): void {
  localStorage.setItem(STORAGE_KEY, id);
}

export function createSessionId(): string {
  const id = `ai-sdk-${crypto.randomUUID()}`;
  localStorage.setItem(STORAGE_KEY, id);
  return id;
}
