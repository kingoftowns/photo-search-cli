import { useQuery } from "@tanstack/react-query";
import { listFaces, type FacesResponse } from "../api/client";

export function useFaces() {
  return useQuery<FacesResponse>({
    queryKey: ["faces"],
    queryFn: listFaces,
    staleTime: 5 * 60_000,
  });
}
