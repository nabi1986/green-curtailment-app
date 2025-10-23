FROM node:20 AS build
WORKDIR /app

# کپی و نصب
COPY package.json package-lock.json* ./
RUN npm install

# کپی سورس
COPY . .

# متغیرهای محیطیِ Vite (در زمان build پر می‌کنیم)
ARG VITE_API_BASE
ARG VITE_WS_BASE
ENV VITE_API_BASE=${VITE_API_BASE}
ENV VITE_WS_BASE=${VITE_WS_BASE}

# ساخت خروجی تولید
RUN npm run build

# سرو با Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
