FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements-worker.txt .
RUN pip install --no-cache-dir -r requirements-worker.txt

# Copy source code
COPY worker.py .

# Expose port and start uvicorn
EXPOSE 8000
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]
