import { useEffect } from "react";
import Box from "@mui/material/Box";
import Dialog from "@mui/material/Dialog";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import Chip from "@mui/material/Chip";
import CloseIcon from "@mui/icons-material/Close";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import { originalUrl, thumbUrl } from "../api/client";
import type { PhotoResult } from "../api/client";

interface LightboxProps {
  results: PhotoResult[];
  index: number;
  onClose: () => void;
  onIndexChange: (next: number) => void;
}

export function Lightbox({ results, index, onClose, onIndexChange }: LightboxProps) {
  const current = results[index];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && index > 0) onIndexChange(index - 1);
      else if (e.key === "ArrowRight" && index < results.length - 1)
        onIndexChange(index + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, results.length, onClose, onIndexChange]);

  if (!current) return null;

  const formattedDate = current.date_taken
    ? new Date(current.date_taken).toLocaleString()
    : null;

  return (
    <Dialog
      open
      onClose={onClose}
      fullScreen
      PaperProps={{
        sx: { bgcolor: "rgba(0,0,0,0.95)", color: "white" },
      }}
    >
      {/* Top bar */}
      <Box
        sx={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          p: 1.5,
          display: "flex",
          alignItems: "center",
          gap: 1,
          zIndex: 2,
          background:
            "linear-gradient(to bottom, rgba(0,0,0,0.6), rgba(0,0,0,0))",
        }}
      >
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography variant="subtitle1" noWrap>
            {current.caption ?? current.file_name}
          </Typography>
          <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap", mt: 0.5 }}>
            {formattedDate && (
              <Chip
                size="small"
                label={formattedDate}
                sx={{ bgcolor: "rgba(255,255,255,0.1)", color: "white" }}
              />
            )}
            {current.location_name && (
              <Chip
                size="small"
                label={current.location_name}
                sx={{ bgcolor: "rgba(255,255,255,0.1)", color: "white" }}
              />
            )}
            {current.camera && (
              <Chip
                size="small"
                label={current.camera}
                sx={{ bgcolor: "rgba(255,255,255,0.1)", color: "white" }}
              />
            )}
            {current.faces.map((p) => (
              <Chip
                key={p}
                size="small"
                label={p}
                sx={{
                  bgcolor: "rgba(144,202,249,0.3)",
                  color: "white",
                }}
              />
            ))}
          </Box>
        </Box>
        <IconButton
          component="a"
          href={originalUrl(current.path_token)}
          target="_blank"
          rel="noreferrer"
          sx={{ color: "white" }}
          aria-label="open original"
        >
          <OpenInNewIcon />
        </IconButton>
        <IconButton onClick={onClose} sx={{ color: "white" }} aria-label="close">
          <CloseIcon />
        </IconButton>
      </Box>

      {/* Image */}
      <Box
        sx={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          p: { xs: 1, md: 4 },
        }}
        onClick={onClose}
      >
        <Box
          component="img"
          src={originalUrl(current.path_token)}
          alt={current.caption ?? current.file_name}
          // Blurred thumbnail shows immediately while the original loads.
          style={{
            backgroundImage: `url(${thumbUrl(current.path_token, 64)})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
          sx={{
            maxWidth: "100%",
            maxHeight: "100%",
            objectFit: "contain",
            borderRadius: 1,
            boxShadow: 8,
          }}
          onClick={(e) => e.stopPropagation()}
        />
      </Box>

      {/* Nav buttons */}
      {index > 0 && (
        <IconButton
          onClick={() => onIndexChange(index - 1)}
          sx={{
            position: "absolute",
            top: "50%",
            left: 8,
            transform: "translateY(-50%)",
            color: "white",
            bgcolor: "rgba(0,0,0,0.3)",
            "&:hover": { bgcolor: "rgba(0,0,0,0.5)" },
          }}
          aria-label="previous"
        >
          <ChevronLeftIcon fontSize="large" />
        </IconButton>
      )}
      {index < results.length - 1 && (
        <IconButton
          onClick={() => onIndexChange(index + 1)}
          sx={{
            position: "absolute",
            top: "50%",
            right: 8,
            transform: "translateY(-50%)",
            color: "white",
            bgcolor: "rgba(0,0,0,0.3)",
            "&:hover": { bgcolor: "rgba(0,0,0,0.5)" },
          }}
          aria-label="next"
        >
          <ChevronRightIcon fontSize="large" />
        </IconButton>
      )}
    </Dialog>
  );
}
