export const apiClient = {
  get(path: string): string {
    return `GET:${path}`;
  },
};

export const clock = {
  now(): string {
    return "2026-03-06T00:00:00Z";
  },
};

export function logger(message: string): string {
  return `[log] ${message}`;
}
