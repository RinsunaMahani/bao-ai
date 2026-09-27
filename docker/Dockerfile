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

# Copy dependency definitions first for optimal layer caching
COPY requirements.txt requirements-ml.txt requirements-pdf.txt /app/

# Install Python dependencies.
#
# requirements-ml.txt, not just requirements.txt: it adds ai-edge-litert,
# the TFLite runtime the language classifier needs (and includes
# requirements.txt itself). Without it the image could not load the LSTM
# at all and ran on the keyword matcher alone, which gets whole sentences
# wrong that the classifier gets right, while the README describes the
# classifier as the default. It is a ~21 MB wheel, not TensorFlow.
#
# requirements-pdf.txt adds pypdf, a pure-Python package of a few hundred
# KB. Without it the upload box accepted a PDF and then said it could not
# read it, which is a bad look for the document-RAG feature the sidebar
# advertises. .txt and .csv uploads already worked.
#
# Speech is deliberately NOT installed: torch and the voice models are
# several GB, and the app degrades to text-only without them rather than
# failing. Build a separate image from requirements-speech.txt for voices.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-ml.txt -r requirements-pdf.txt

# Copy the entire project repository context into /app
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Lets Docker, and compose's restart policy, tell a running container from
# a healthy one. /_stcore/health is Streamlit's own readiness route. The
# start period covers the first load building the pipeline (classifier,
# knowledge base, detector bundle), which takes several seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Execute Streamlit server pointing to bao/ui/streamlit_app.py
CMD ["streamlit", "run", "bao/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]