import { useState } from "react";
import Box from "@mui/material/Box";
import ImageList from "@mui/material/ImageList";
import ImageListItem from "@mui/material/ImageListItem";
import ImageListItemBar from "@mui/material/ImageListItemBar";
import IconButton from "@mui/material/IconButton";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import Skeleton from "@mui/material/Skeleton";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import { thumbUrl } from "../api/client";
import type { PhotoResult } from "../api/client";
import { Lightbox } from "./Lightbox";

interface ResultsGridProps {
  results: PhotoResult[];
  query: string;
  active: boolean;
  loading?: boolean;
  error: string | null;
}

function formatDate(iso: string | null): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return iso;
  }
}

export function ResultsGrid({ results, query, active, loading, error }: ResultsGridProps) {
  const theme = useTheme();
  const upLg = useMediaQuery(theme.breakpoints.up("lg"));
  const upMd = useMediaQuery(theme.breakpoints.up("md"));
  const upSm = useMediaQuery(theme.breakpoints.up("sm"));
  const cols = upLg ? 5 : upMd ? 4 : upSm ? 3 : 2;

  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!active) {
    return (
      <Box
        sx={{
          mt: 8,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1,
          color: "text.secondary",
        }}
      >
        <Typography variant="h6">Start typing or pick a filter to search your photos</Typography>
        <Typography variant="body2">
          Try things like "sunset at the beach", "birthday party", or "hiking in 2024".
        </Typography>
      </Box>
    );
  }

  if (loading) {
    return (
      <ImageList variant="masonry" cols={cols} gap={8} sx={{ mt: 1 }}>
        {Array.from({ length: 12 }).map((_, i) => (
          <ImageListItem key={i}>
            <Skeleton
              variant="rectangular"
              sx={{ borderRadius: 1 }}
              height={180 + (i % 4) * 40}
            />
          </ImageListItem>
        ))}
      </ImageList>
    );
  }

  if (results.length === 0) {
    return (
      <Typography sx={{ mt: 4 }} color="text.secondary" align="center">
        {query ? `No results for "${query}".` : "No photos match the current filters."}
      </Typography>
    );
  }

  return (
    <>
      <ImageList variant="masonry" cols={cols} gap={8} sx={{ mt: 1 }}>
        {results.map((r, i) => (
          <ImageListItem
            key={r.path_token}
            onClick={() => setLightboxIdx(i)}
            sx={{
              cursor: "pointer",
              borderRadius: 1,
              overflow: "hidden",
              "&:hover .MuiImageListItemBar-root": { opacity: 1 },
              "& img": {
                transition: "transform 200ms ease",
              },
              "&:hover img": {
                transform: "scale(1.02)",
              },
            }}
          >
            <img
              src={thumbUrl(r.path_token, 600)}
              alt={r.caption ?? r.file_name}
              loading="lazy"
            />
            <ImageListItemBar
              sx={{
                background:
                  "linear-gradient(to top, rgba(0,0,0,0.75) 0%, rgba(0,0,0,0.3) 50%, rgba(0,0,0,0) 100%)",
                opacity: 0,
                transition: "opacity 150ms ease",
              }}
              title={
                <Typography variant="body2" noWrap>
                  {r.caption ?? r.file_name}
                </Typography>
              }
              subtitle={
                <Box
                  sx={{
                    display: "flex",
                    gap: 0.5,
                    flexWrap: "wrap",
                    alignItems: "center",
                  }}
                >
                  {r.date_taken && (
                    <Chip
                      size="small"
                      label={formatDate(r.date_taken)}
                      variant="filled"
                      sx={{ bgcolor: "rgba(255,255,255,0.15)", color: "white" }}
                    />
                  )}
                  {r.location_name && (
                    <Chip
                      size="small"
                      label={r.location_name}
                      variant="filled"
                      sx={{ bgcolor: "rgba(255,255,255,0.15)", color: "white" }}
                    />
                  )}
                  {r.faces.slice(0, 3).map((p) => (
                    <Chip
                      key={p}
                      size="small"
                      label={p}
                      variant="filled"
                      sx={{ bgcolor: "rgba(144,202,249,0.35)", color: "white" }}
                    />
                  ))}
                </Box>
              }
              actionIcon={
                <IconButton
                  sx={{ color: "rgba(255,255,255,0.7)" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setLightboxIdx(i);
                  }}
                >
                  <InfoOutlinedIcon />
                </IconButton>
              }
            />
          </ImageListItem>
        ))}
      </ImageList>

      {lightboxIdx !== null && (
        <Lightbox
          results={results}
          index={lightboxIdx}
          onClose={() => setLightboxIdx(null)}
          onIndexChange={setLightboxIdx}
        />
      )}
    </>
  );
}
