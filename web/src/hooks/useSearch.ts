import { useQuery } from "@tanstack/react-query";
import { search, type SearchParams, type SearchResponse } from "../api/client";

export function useSearch(params: SearchParams, enabled = true) {
  return useQuery<SearchResponse>({
    queryKey: ["search", params],
    queryFn: () => search(params),
    enabled,
    placeholderData: (prev) => prev,
  });
}
