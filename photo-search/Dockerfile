# photo-search API image.
#
# Packages the Python CLI + FastAPI service into a single container. The
# same image can be reused as a k8s Job to run `photo-search index` etc.,
# though in production we run the API via uvicorn as CMD below.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for Pillow/pillow-heif, onnxruntime, opencv-headless,
# and psycopg2 runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libheif1 \
      libde265-0 \
      libjpeg62-turbo \
      libpng16-16 \
      libtiff6 \
      libopenblas0 \
      libglib2.0-0 \
      libgl1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY photo_search ./photo_search
COPY api ./api
COPY scripts ./scripts

RUN pip install --upgrade pip \
 && pip install .

# Writable thumbnail cache directory (mount a Volume here in k8s).
RUN mkdir -p /var/cache/photo-thumbs
ENV PHOTO_SEARCH_THUMB_CACHE=/var/cache/photo-thumbs

EXPOSE 8000

# Config file location inside the container. The photo-api Helm chart
# mounts a ConfigMap at /app/config.yaml.
WORKDIR /app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
