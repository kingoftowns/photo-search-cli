# Multi-stage build: compile the React app with Node, then serve the
# static bundle with nginx. In two-ingress deployments the browser calls
# the API host directly via CORS, so nginx only serves static files here.

FROM node:20-alpine AS build
WORKDIR /app

# Baked into the bundle at build time. Defaults to same-origin for dev;
# k8s builds pass e.g. https://api.photos.k8s.blacktoaster.com.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=${VITE_API_BASE}

COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
