import { createTheme, type PaletteMode } from "@mui/material";

export function buildTheme(mode: PaletteMode) {
  const isDark = mode === "dark";
  return createTheme({
    palette: {
      mode,
      primary: { main: isDark ? "#90caf9" : "#1976d2" },
      secondary: { main: isDark ? "#ce93d8" : "#9c27b0" },
      background: isDark
        ? { default: "#0e0e10", paper: "#17171a" }
        : { default: "#fafafa", paper: "#ffffff" },
    },
    shape: { borderRadius: 12 },
    typography: {
      fontFamily: 'Roboto, system-ui, -apple-system, "Segoe UI", sans-serif',
    },
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
            backdropFilter: "blur(8px)",
            backgroundColor: isDark
              ? "rgba(23, 23, 26, 0.75)"
              : "rgba(255, 255, 255, 0.8)",
          },
        },
      },
      MuiButton: { defaultProps: { disableElevation: true } },
      MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
    },
  });
}
