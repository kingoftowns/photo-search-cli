import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import InputBase from "@mui/material/InputBase";
import IconButton from "@mui/material/IconButton";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import SearchIcon from "@mui/icons-material/Search";
import ClearIcon from "@mui/icons-material/Clear";
import PlaceIcon from "@mui/icons-material/Place";
import CalendarMonthIcon from "@mui/icons-material/CalendarMonth";
import type { LocationSuggestion } from "../api/client";

export interface SearchFilters {
  q: string;
  people: string[];
  location: LocationSuggestion | null;
  year?: number;
  after?: string;
  before?: string;
}

interface SearchBarProps {
  value: SearchFilters;
  onChange: (next: SearchFilters) => void;
  busy?: boolean;
  // Year parsed out of the current query text, for display feedback only.
  extractedYear?: number | null;
}

export function SearchBar({ value, onChange, busy, extractedYear }: SearchBarProps) {
  const [local, setLocal] = useState(value.q);

  // Keep local input in sync if filters change from outside (e.g. clearing
  // person chip shouldn't clobber user typing).
  useEffect(() => {
    setLocal(value.q);
  }, [value.q]);

  // 300ms debounce before committing the query upstream.
  useEffect(() => {
    const id = setTimeout(() => {
      if (local !== value.q) onChange({ ...value, q: local });
    }, 300);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [local]);

  const removePerson = (label: string) =>
    onChange({ ...value, people: value.people.filter((p) => p !== label) });
  const clearQuery = () => {
    setLocal("");
    onChange({ ...value, q: "" });
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
      <Paper
        elevation={0}
        sx={{
          display: "flex",
          alignItems: "center",
          px: 2,
          py: 0.5,
          borderRadius: 999,
          border: 1,
          borderColor: "divider",
          maxWidth: 900,
          mx: "auto",
          width: "100%",
        }}
      >
        <SearchIcon sx={{ color: "text.secondary", mr: 1 }} />
        <InputBase
          sx={{ flex: 1, fontSize: 16 }}
          placeholder="Describe what you're looking for…"
          value={local}
          onChange={(e) => setLocal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onChange({ ...value, q: local });
          }}
          inputProps={{ "aria-label": "search photos" }}
        />
        {busy && <CircularProgress size={18} sx={{ mr: 1 }} />}
        {local && (
          <IconButton size="small" onClick={clearQuery} aria-label="clear">
            <ClearIcon fontSize="small" />
          </IconButton>
        )}
      </Paper>
      {(value.people.length > 0 || value.location || extractedYear != null) && (
        <Box
          sx={{
            display: "flex",
            flexWrap: "wrap",
            gap: 0.5,
            justifyContent: "center",
          }}
        >
          {value.location && (
            <Chip
              icon={<PlaceIcon fontSize="small" />}
              label={value.location.display}
              onDelete={() => onChange({ ...value, location: null })}
              color="primary"
              variant="outlined"
              size="small"
            />
          )}
          {extractedYear != null && (
            <Chip
              icon={<CalendarMonthIcon fontSize="small" />}
              label={extractedYear}
              color="primary"
              variant="outlined"
              size="small"
            />
          )}
          {value.people.map((p) => (
            <Chip
              key={p}
              label={p}
              onDelete={() => removePerson(p)}
              color="primary"
              variant="outlined"
              size="small"
            />
          ))}
        </Box>
      )}
    </Box>
  );
}
