import { useMemo, useState } from "react";
import Box from "@mui/material/Box";
import { AppShell } from "./components/AppShell";
import { SearchBar, type SearchFilters } from "./components/SearchBar";
import { ResultsGrid } from "./components/ResultsGrid";
import { useSearch } from "./hooks/useSearch";

export default function App() {
  const [filters, setFilters] = useState<SearchFilters>({
    q: "",
    people: [],
    location: null,
  });

  // Pull a standalone 4-digit year (1900-2100) out of the query so it can be
  // pushed down as a structured `year` filter instead of matched lexically.
  const parsedQuery = useMemo(() => {
    const raw = filters.q.trim();
    const match = raw.match(/\b(\d{4})\b/);
    if (match && match.index !== undefined) {
      const y = parseInt(match[1], 10);
      if (y >= 1900 && y <= 2100) {
        const stripped = (
          raw.slice(0, match.index) + raw.slice(match.index + match[0].length)
        )
          .replace(/\s+/g, " ")
          .trim();
        return { q: stripped, year: y };
      }
    }
    return { q: raw, year: null as number | null };
  }, [filters.q]);

  const params = useMemo(
    () => ({
      q: parsedQuery.q,
      people: filters.people,
      city: filters.location?.city ?? null,
      region: filters.location?.region ?? null,
      country_code: filters.location?.country_code ?? null,
      year: parsedQuery.year ?? filters.year ?? null,
      after: filters.after ?? null,
      before: filters.before ?? null,
      top: 60,
    }),
    [parsedQuery, filters],
  );

  const hasFilters =
    params.people.length > 0 ||
    !!params.city ||
    !!params.region ||
    !!params.country_code ||
    params.year != null ||
    !!params.after ||
    !!params.before;
  const enabled = params.q.length > 0 || hasFilters;

  const { data, isFetching, error } = useSearch(params, enabled);

  return (
    <AppShell
      selectedPeople={filters.people}
      onChangePeople={(people) => setFilters((f) => ({ ...f, people }))}
      selectedLocation={filters.location}
      onChangeLocation={(location) => setFilters((f) => ({ ...f, location }))}
    >
      <Box
        sx={{
          position: "sticky",
          top: 0,
          zIndex: 5,
          pt: 2,
          pb: 2,
          px: 3,
          backdropFilter: "blur(8px)",
          backgroundColor: (t) => t.palette.background.default + "cc",
        }}
      >
        <SearchBar
          value={filters}
          onChange={setFilters}
          busy={isFetching}
          extractedYear={parsedQuery.year}
        />
      </Box>

      <Box sx={{ px: 3, pb: 6 }}>
        <ResultsGrid
          results={data?.results ?? []}
          query={params.q}
          active={enabled}
          error={error instanceof Error ? error.message : null}
          loading={isFetching && !data}
        />
      </Box>
    </AppShell>
  );
}
