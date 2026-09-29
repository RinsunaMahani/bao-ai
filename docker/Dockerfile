# Base image pinned by digest. A tag such as 3.11-slim is moved to a new
# image every time it is rebuilt, so the same Dockerfile could produce a
# different image from one week to the next. This is the multi-platform
# index for python:3.11-slim as pushed on 2026-09-25; Dependabot
# (.github/dependabot.yml) proposes newer digests as they are published.
FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

WORKDIR /app

# PYTHONPATH so absolute imports (bao.*) resolve. No .pyc files and no
# stdout buffering, as before. No pip cache or version check in a layer
# that is built once and never updated in place.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# No apt layer. The image used to install build-essential and curl: every
# dependency here ships a prebuilt wheel for Python 3.11, so the compiler
# toolchain was never used, and it was the largest source of packages -
# and so of CVE reports - in the image. curl only ran the health check,
# which uses Python now.
COPY requirements.txt requirements-ml.txt requirements-pdf.txt /app/

# requirements-ml.txt adds ai-edge-litert, the TFLite runtime the language
# classifier needs (and includes requirements.txt itself). Without it the
# image could not load the LSTM and ran on the keyword matcher alone. It is
# a ~21 MB wheel, not TensorFlow.
#
# requirements-pdf.txt adds pypdf, a pure-Python package of a few hundred
# KB. Without it the upload box accepted a PDF and then said it could not
# read it. .txt and .csv uploads already worked.
#
# Speech is deliberately NOT installed: torch and the voice models are
# several GB, and the app degrades to text-only without them rather than
# failing. Build a separate image from requirements-speech.txt for voices.
RUN pip install -r requirements-ml.txt -r requirements-pdf.txt

# The application files stay owned by root: the user the app runs as can
# read them but not change them, so a compromised process cannot rewrite
# the code the next start would run.
COPY . /app

# An unprivileged user with no login shell. Root inside a container is
# still root on the files it can reach, and nothing here needs it.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin bao
USER bao

EXPOSE 8501

# Lets Docker, and compose's restart policy, tell a running container from
# a healthy one. /_stcore/health is Streamlit's own readiness route. The
# start period covers the first load building the pipeline (classifier,
# knowledge base, detector bundle), which takes several seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

# 0.0.0.0 inside the container: its own localhost is unreachable from the
# host. What the HOST exposes is decided by the port binding, which
# docker-compose.yml limits to 127.0.0.1. The other server settings come
# from .streamlit/config.toml.
CMD ["streamlit", "run", "bao/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
