# Dockerfile for NourishNet AI
# Builds a CPU-based image. Note: installing full TF/PyTorch may increase image size significantly.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps for some Python packages and torch torchvision (may need extra on some platforms)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Ensure scripts are executable
RUN chmod +x /app/scripts/bootstrap_models.sh || true

EXPOSE 8000

# Default command: bootstrap models (if needed) then start the API
CMD ["bash", "-c", "./scripts/bootstrap_models.sh && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]
