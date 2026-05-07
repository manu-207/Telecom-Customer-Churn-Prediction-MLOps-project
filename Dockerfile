FROM python:3.11-slim

LABEL maintainer="mlops-team"
LABEL description="ChurnGuard - Multi-Model Registry with Automated Promotion"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for the Flask prediction API
EXPOSE 5001

# Default command: run the prediction API
CMD ["python", "app.py"]
