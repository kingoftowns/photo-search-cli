import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { PaletteMode } from "@mui/material";

interface ThemeModeCtx {
  mode: PaletteMode;
  toggle: () => void;
}

const Ctx = createContext<ThemeModeCtx | null>(null);

const STORAGE_KEY = "photo-search.themeMode";

function loadInitial(): PaletteMode {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* ignore */
  }
  return "dark";
}

export function ThemeModeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<PaletteMode>(loadInitial);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      /* ignore */
    }
  }, [mode]);

  const value = useMemo<ThemeModeCtx>(
    () => ({
      mode,
      toggle: () => setMode((m) => (m === "dark" ? "light" : "dark")),
    }),
    [mode],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useThemeMode(): ThemeModeCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useThemeMode must be used inside ThemeModeProvider");
  return v;
}
