import { useEffect, useState } from "react";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import Box from "@mui/material/Box";
import ListSubheader from "@mui/material/ListSubheader";
import Typography from "@mui/material/Typography";
import { useLocations } from "../hooks/useLocations";
import type { LocationSuggestion } from "../api/client";

interface LocationFilterProps {
  selected: LocationSuggestion | null;
  onChange: (next: LocationSuggestion | null) => void;
}

export function LocationFilter({ selected, onChange }: LocationFilterProps) {
  const [input, setInput] = useState("");
  const [prefix, setPrefix] = useState("");

  // Debounce the query so we don't hit the API on every keystroke.
  useEffect(() => {
    const id = setTimeout(() => setPrefix(input), 250);
    return () => clearTimeout(id);
  }, [input]);

  const { data, isFetching } = useLocations(prefix);
  const options = data?.locations ?? [];

  return (
    <Box>
      <ListSubheader sx={{ bgcolor: "transparent", lineHeight: "32px" }}>
        Location
      </ListSubheader>
      <Box sx={{ px: 2, pb: 1 }}>
        <Autocomplete<LocationSuggestion, false, false, false>
          size="small"
          value={selected}
          onChange={(_, next) => onChange(next)}
          inputValue={input}
          onInputChange={(_, next) => setInput(next)}
          options={options}
          loading={isFetching}
          getOptionLabel={(o) => o.display}
          isOptionEqualToValue={(a, b) => a.display === b.display}
          // Server already filters by prefix; don't re-filter on the client.
          filterOptions={(x) => x}
          noOptionsText={
            prefix.length === 0 ? "Start typing…" : "No matches"
          }
          renderOption={(props, option) => (
            <li {...props} key={option.display}>
              <Box sx={{ display: "flex", flexDirection: "column" }}>
                <span>{option.display}</span>
                <Typography variant="caption" color="text.secondary">
                  {option.photo_count} photo
                  {option.photo_count === 1 ? "" : "s"}
                </Typography>
              </Box>
            </li>
          )}
          renderInput={(params) => (
            <TextField {...params} placeholder="Search locations…" />
          )}
        />
      </Box>
    </Box>
  );
}
