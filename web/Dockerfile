# Multi-stage build: compile the React app with Node, then serve the
# static bundle with nginx. The nginx container also proxies /api, /thumbs,
# and /originals to the photo-api service so the browser only talks to
# one origin.

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template

# The nginx image's entrypoint runs envsubst on templates using any
# variables we set, so the Helm chart can point at the right upstream by
# setting API_UPSTREAM.
ENV API_UPSTREAM=http://photo-api:8000

EXPOSE 80
