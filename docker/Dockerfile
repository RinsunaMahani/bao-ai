# Base Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ensure /app is in PYTHONPATH so absolute imports (bao.*) resolve cleanly
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition first for optimal layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project repository context into /app
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Execute Streamlit server pointing to bao/ui/streamlit_app.py
CMD ["streamlit", "run", "bao/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]