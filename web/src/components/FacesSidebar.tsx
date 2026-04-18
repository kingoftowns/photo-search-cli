import Box from "@mui/material/Box";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import ListSubheader from "@mui/material/ListSubheader";
import Checkbox from "@mui/material/Checkbox";
import Chip from "@mui/material/Chip";
import Skeleton from "@mui/material/Skeleton";
import Typography from "@mui/material/Typography";
import PeopleOutlineIcon from "@mui/icons-material/PeopleOutline";
import { useFaces } from "../hooks/useFaces";

interface FacesSidebarProps {
  selected: string[];
  onChange: (next: string[]) => void;
}

export function FacesSidebar({ selected, onChange }: FacesSidebarProps) {
  const { data, isLoading, error } = useFaces();

  return (
    <Box sx={{ overflowY: "auto" }}>
      <List
        dense
        subheader={
          <ListSubheader sx={{ bgcolor: "transparent", lineHeight: "32px" }}>
            People
          </ListSubheader>
        }
      >
        <ListItemButton
          selected={selected.length === 0}
          onClick={() => onChange([])}
        >
          <ListItemIcon>
            <PeopleOutlineIcon />
          </ListItemIcon>
          <ListItemText primary="Everyone" />
        </ListItemButton>

        {isLoading &&
          Array.from({ length: 5 }).map((_, i) => (
            <ListItemButton key={i} disabled>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Skeleton variant="rectangular" width={18} height={18} />
              </ListItemIcon>
              <ListItemText primary={<Skeleton width={120} />} />
            </ListItemButton>
          ))}

        {error && (
          <Typography variant="body2" color="error" sx={{ px: 2, py: 1 }}>
            Couldn't load faces
          </Typography>
        )}

        {data?.faces.map((f) => {
          const isSelected = selected.includes(f.label);
          const toggle = () =>
            onChange(
              isSelected
                ? selected.filter((x) => x !== f.label)
                : [...selected, f.label],
            );
          return (
            <ListItemButton key={f.label} selected={isSelected} onClick={toggle}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Checkbox
                  edge="start"
                  checked={isSelected}
                  tabIndex={-1}
                  disableRipple
                  size="small"
                />
              </ListItemIcon>
              <ListItemText primary={f.display_name} />
            </ListItemButton>
          );
        })}

        {data && data.faces.length === 0 && (
          <Box sx={{ px: 2, py: 1 }}>
            <Chip
              size="small"
              label="No labeled faces yet"
              variant="outlined"
            />
          </Box>
        )}
      </List>
    </Box>
  );
}
