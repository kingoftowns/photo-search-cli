// Typed client for the photo-search FastAPI service.
//
// In dev Vite proxies /api, /thumbs, /originals to the Python server.
// In prod the nginx container does the same.

const BASE = import.meta.env.VITE_API_BASE ?? "";

export interface PhotoResult {
  file_path: string;
  file_name: string;
  score: number;
  caption: string | null;
  faces: string[];
  date_taken: string | null;
  location_name: string | null;
  camera: string | null;
  path_token: string;
  thumb_url: string;
  original_url: string;
}

export interface SearchResponse {
  query: string;
  count: number;
  results: PhotoResult[];
}

export interface FaceIdentity {
  label: string;
  display_name: string;
  sample_count: number;
}

export interface FacesResponse {
  count: number;
  faces: FaceIdentity[];
}

export interface LocationSuggestion {
  display: string;
  city: string | null;
  region: string | null;
  country_code: string | null;
  photo_count: number;
}

export interface LocationsResponse {
  count: number;
  locations: LocationSuggestion[];
}

export interface SearchParams {
  q?: string;
  people?: string[];
  city?: string | null;
  region?: string | null;
  country_code?: string | null;
  year?: number | null;
  after?: string | null;
  before?: string | null;
  top?: number;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body || path}`);
  }
  return (await res.json()) as T;
}

export function search(params: SearchParams): Promise<SearchResponse> {
  const qs = new URLSearchParams();
  if (params.q) qs.set("q", params.q);
  for (const p of params.people ?? []) qs.append("person", p);
  if (params.city) qs.set("city", params.city);
  if (params.region) qs.set("region", params.region);
  if (params.country_code) qs.set("country_code", params.country_code);
  if (params.year != null) qs.set("year", String(params.year));
  if (params.after) qs.set("after", params.after);
  if (params.before) qs.set("before", params.before);
  qs.set("top", String(params.top ?? 60));
  return getJson<SearchResponse>(`/api/search?${qs.toString()}`);
}

export function listFaces(): Promise<FacesResponse> {
  return getJson<FacesResponse>(`/api/faces`);
}

export function listLocations(
  prefix: string,
  limit = 20,
): Promise<LocationsResponse> {
  const qs = new URLSearchParams({ prefix, limit: String(limit) });
  return getJson<LocationsResponse>(`/api/locations?${qs.toString()}`);
}

export function thumbUrl(token: string, size = 400): string {
  return `${BASE}/thumbs/${token}?size=${size}`;
}

export function originalUrl(token: string): string {
  return `${BASE}/originals/${token}`;
}
