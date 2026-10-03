# Hugging Face Spaces (Docker SDK) image for the pyKinaXe backend.
# HF expects the app to listen on 0.0.0.0:7860.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYKINAXE_WEB_HOST=0.0.0.0 \
    PYKINAXE_WEB_PORT=7860 \
    PYKINAXE_WEB_RUNTIME_ROOT=/app/webapp/runtime \
    JOB_RETENTION_HOURS=1 \
    JOB_MAX_AGE_SECONDS=3600

# Add build tools for compiling Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libgomp1 libstdc++6 libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# System libs required by opencv-python-headless / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libstdc++6 \
        libfontconfig1 \
        libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install gunicorn

# Copy the project.
COPY . .

EXPOSE 7860

# Use gunicorn for stability on HF. Threaded worker so background job thread
# (the reaper) and the ThreadPoolExecutor share process state.
CMD ["gunicorn", \
     "--chdir", "webapp", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "0", \
     "pykinaxe_webapp:app"]
