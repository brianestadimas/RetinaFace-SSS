# ---------- GPU BASE ----------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps (OpenCV runtime libs only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential gcc g++ make \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# App
COPY main.py /app/main.py

# Runtime env
ENV FACE_DET_SIZE=480,480
ENV ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider

EXPOSE 8000

# Keep workers=1 so a single GPU context is shared safely;
# scale with multiple containers instead of multiple workers.
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--workers=1"]
