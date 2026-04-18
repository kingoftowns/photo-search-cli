import { useState } from "react";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Tooltip from "@mui/material/Tooltip";
import MenuIcon from "@mui/icons-material/Menu";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import PhotoLibraryIcon from "@mui/icons-material/PhotoLibrary";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import { FacesSidebar } from "./FacesSidebar";
import { LocationFilter } from "./LocationFilter";
import { useThemeMode } from "../theme/ThemeModeProvider";
import type { LocationSuggestion } from "../api/client";

const DRAWER_WIDTH = 260;

interface AppShellProps {
  children: React.ReactNode;
  selectedPeople: string[];
  onChangePeople: (next: string[]) => void;
  selectedLocation: LocationSuggestion | null;
  onChangeLocation: (next: LocationSuggestion | null) => void;
}

export function AppShell({
  children,
  selectedPeople,
  onChangePeople,
  selectedLocation,
  onChangeLocation,
}: AppShellProps) {
  const theme = useTheme();
  const isDesktop = useMediaQuery(theme.breakpoints.up("md"));
  const [mobileOpen, setMobileOpen] = useState(false);
  const { mode, toggle } = useThemeMode();

  const drawerContent = (
    <>
      <LocationFilter
        selected={selectedLocation}
        onChange={onChangeLocation}
      />
      <FacesSidebar selected={selectedPeople} onChange={onChangePeople} />
    </>
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100%" }}>
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          zIndex: (t) => t.zIndex.drawer + 1,
          borderBottom: 1,
          borderColor: "divider",
        }}
      >
        <Toolbar>
          {!isDesktop && (
            <IconButton
              color="inherit"
              edge="start"
              onClick={() => setMobileOpen((v) => !v)}
              sx={{ mr: 1 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          <PhotoLibraryIcon sx={{ mr: 1, color: "primary.main" }} />
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 500 }}>
            photo-search
          </Typography>
          <Tooltip title={mode === "dark" ? "Light mode" : "Dark mode"}>
            <IconButton color="inherit" onClick={toggle}>
              {mode === "dark" ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      {isDesktop ? (
        <Drawer
          variant="permanent"
          sx={{
            width: DRAWER_WIDTH,
            flexShrink: 0,
            "& .MuiDrawer-paper": {
              width: DRAWER_WIDTH,
              boxSizing: "border-box",
              borderRight: 1,
              borderColor: "divider",
            },
          }}
        >
          <Toolbar />
          {drawerContent}
        </Drawer>
      ) : (
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          ModalProps={{ keepMounted: true }}
          sx={{
            "& .MuiDrawer-paper": { width: DRAWER_WIDTH },
          }}
        >
          <Toolbar />
          {drawerContent}
        </Drawer>
      )}

      <Box component="main" sx={{ flexGrow: 1, minWidth: 0 }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}
