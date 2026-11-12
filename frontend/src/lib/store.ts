import { create } from "zustand";

type State = {
  model: string;
  traceId: string | null;
  theme: "light" | "dark";
  setModel: (m: string) => void;
  setTrace: (t: string | null) => void;
  toggleTheme: () => void;
};

const initialTheme = (localStorage.getItem("theme") as "light" | "dark") ||
  (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");
document.documentElement.dataset.theme = initialTheme;

export const useApp = create<State>((set) => ({
  model: localStorage.getItem("model") || "qwen3.5-0.8b",
  traceId: localStorage.getItem("trace"),
  theme: initialTheme,
  setModel: (model) => { localStorage.setItem("model", model); set({ model }); },
  setTrace: (traceId) => { traceId ? localStorage.setItem("trace", traceId) : localStorage.removeItem("trace"); set({ traceId }); },
  toggleTheme: () => set((s) => {
    const theme = s.theme === "light" ? "dark" : "light";
    localStorage.setItem("theme", theme);
    document.documentElement.dataset.theme = theme;
    return { theme };
  }),
}));
