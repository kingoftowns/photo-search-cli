import { useQuery } from "@tanstack/react-query";
import { listLocations, type LocationsResponse } from "../api/client";

export function useLocations(prefix: string, enabled = true) {
  return useQuery<LocationsResponse>({
    queryKey: ["locations", prefix],
    queryFn: () => listLocations(prefix),
    enabled,
    staleTime: 5 * 60_000,
    placeholderData: (prev) => prev,
  });
}
