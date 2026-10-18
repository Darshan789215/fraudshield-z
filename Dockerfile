FROM node:22-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
ENV NODE_ENV=production
ENV PORT=3000
ENV FRAUDSHIELD_BIND_HOST=0.0.0.0
ENV FRAUDSHIELD_DATA_DIR=/var/lib/fraudshield
VOLUME ["/var/lib/fraudshield"]
EXPOSE 3000
CMD ["node", "server/index.js"]
