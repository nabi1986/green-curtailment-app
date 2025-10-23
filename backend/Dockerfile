FROM python:3.11-slim

WORKDIR /app

# کپی فایل نیازمندی‌ها
COPY requirements.txt .

# نصب FastAPI، Uvicorn و بقیه پکیج‌ها
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]" fastapi

# کپی کد
COPY main.py .

EXPOSE 8000

# اجرای سرور
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
