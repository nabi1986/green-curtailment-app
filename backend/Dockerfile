FROM python:3.11-slim

# optional but helpful
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Show what exists in the build context (for Render logs)
# If these fail, مسیر/کانتکست مشکل دارد.
COPY . /app/
RUN echo "=== LS /app ===" && ls -la /app && \
    echo "=== Show requirements.txt ===" && sed -n '1,120p' /app/requirements.txt

# Install deps
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir "uvicorn[standard]" fastapi

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
