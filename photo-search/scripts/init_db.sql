-- Photo-search PostgreSQL schema.
-- Idempotent: safe to run multiple times via IF NOT EXISTS guards.

CREATE TABLE IF NOT EXISTS photos (
    file_path       TEXT PRIMARY KEY,
    file_name       TEXT NOT NULL,
    caption         TEXT,
    date_taken      TIMESTAMPTZ,
    gps_lat         DOUBLE PRECISION,
    gps_lon         DOUBLE PRECISION,
    location_name   TEXT,
    camera          TEXT,
    file_type       TEXT,
    file_size_bytes BIGINT,
    width           INTEGER,
    height          INTEGER,
    caption_model   TEXT,
    embedding_model TEXT,
    indexed_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS face_identities (
    label              TEXT PRIMARY KEY,
    display_name       TEXT NOT NULL,
    centroid_embedding BYTEA,
    sample_count       INTEGER DEFAULT 0,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS photo_faces (
    id              SERIAL PRIMARY KEY,
    photo_file_path TEXT REFERENCES photos(file_path) ON DELETE CASCADE,
    face_label      TEXT,
    confidence      REAL,
    similarity      REAL,
    bbox_x          REAL,
    bbox_y          REAL,
    bbox_w          REAL,
    bbox_h          REAL,
    embedding       BYTEA,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS indexing_status (
    file_path        TEXT PRIMARY KEY,
    exif_extracted   BOOLEAN DEFAULT FALSE,
    faces_extracted  BOOLEAN DEFAULT FALSE,
    faces_classified BOOLEAN DEFAULT FALSE,
    captioned        BOOLEAN DEFAULT FALSE,
    embedded         BOOLEAN DEFAULT FALSE,
    error            TEXT,
    last_updated     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_photo_faces_label ON photo_faces(face_label);
CREATE INDEX IF NOT EXISTS idx_photo_faces_photo ON photo_faces(photo_file_path);
CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date_taken);
CREATE INDEX IF NOT EXISTS idx_indexing_status_incomplete ON indexing_status(embedded) WHERE embedded = FALSE;
