# Face Verification API (FastAPI + InsightFace)

FastAPI service for face enroll & verification using InsightFace (`buffalo_l`).  
Supports:
- Enroll via **JSON (base64)** or **multipart files**
- Verify via **single HTTP POST** (multipart or JSON) and **WebSocket stream** (binary or JSON)
- Clear per-employee embeddings
- Swagger UI

---

## Endpoints

- `POST /face/enroll` — JSON base64 enroll
- `POST /face/enroll-files` — multipart file enroll
- `POST /face/verify` — verify one frame (multipart `frame`/`image` or JSON data URL)
- `WEBSOCKET /face/verify` — stream frames (binary bytes or JSON data URL)
- `POST /face/verify-image` — JSON base64 verify
- `POST /face/clear` — clear embeddings
- `GET /health` — health check

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)  
ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## GPU Setup

Use if you want speed and have an NVIDIA GPU.

**Image requirements:**
- `onnxruntime-gpu` in requirements.txt
- Base image like `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`

### 1. Install NVIDIA driver on host

```bash
nvidia-smi   # should print your GPU details
```

### 2. Install NVIDIA Container Toolkit (Ubuntu example)

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
 | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
 | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
 | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Test GPU passthrough

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 4. Build & Run (GPU)

```bash
sudo docker build -t face-api-gpu .
sudo docker run --rm -it --gpus all -p 8000:8000 \
  -e ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
  -e FACE_SIM_THRESHOLD=0.75 \
  -v $HOME/.insightface:/root/.insightface \
  face-api-gpu
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Troubleshooting

- **Error:** could not select device driver "" with capabilities: [[gpu]]
  - NVIDIA Container Toolkit not set up → follow the GPU Setup steps.
- **Old Docker (<19.03):**
  ```bash
  sudo docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ...
  ```
- **Warning:** The NVIDIA Driver was not detected
  - Install the host driver (`nvidia-smi` must work), then restart Docker.
- **Container exits during startup**
  - Try CPU first:
    ```bash
    sudo docker run --rm -it -p 8000:8000 \
      -e ORT_PROVIDERS=CPUExecutionProvider \
      -v $HOME/.insightface:/root/.insightface \
      face-api-gpu
    ```
- **Permissions error on Docker socket**
  - Add user to docker group:
    ```bash
    sudo usermod -aG docker $USER
    # log out/in or reboot
    ```
  - Or use `sudo docker ...`


---


## Open Swagger
```bash
http://localhost:8000/docs/
```

---

## Curl Examples

### Enroll (files)

```bash
curl -X POST "http://localhost:8000/face/enroll-files" \
  -F "employee_id=1" \
  -F "files=@/path/to/3x4.jpg" \
  -F "files=@/path/to/another.jpg"
```

### Enroll (base64 JSON)

```bash
IMG_B64="$(base64 -w0 /path/to/3x4.jpg)"
curl -X POST "http://localhost:8000/face/enroll" \
  -H "Content-Type: application/json" \
  -d "{\"employee_id\":\"1\",\"images\":[\"data:image/jpeg;base64,$IMG_B64\"]}"
```

### Verify (single, multipart file)

```bash
curl -X POST "http://localhost:8000/face/verify?employee_id=1&threshold=0.75" \
  -F "frame=@/path/to/capture.jpg"
```

### Verify (single, JSON data URL)

```bash
IMG_B64="$(base64 -w0 /path/to/capture.jpg)"
curl -X POST "http://localhost:8000/face/verify?employee_id=1&threshold=0.75" \
  -H "Content-Type: application/json" \
  -d "{\"frame\":\"data:image/jpeg;base64,$IMG_B64\"}"
```

### Verify (image-only JSON)

```bash
IMG_B64="$(base64 -w0 /path/to/capture.jpg)"
curl -X POST "http://localhost:8000/face/verify-image" \
  -H "Content-Type: application/json" \
  -d "{\"employee_id\":\"1\",\"image\":\"data:image/jpeg;base64,$IMG_B64\",\"threshold\":0.75}"
```

### Clear embeddings

```bash
curl -X POST "http://localhost:8000/face/clear" \
  -H "Content-Type: application/json" \
  -d '{"employee_id":"1"}'
```
---

## Dev (local, no Docker)

```bash
pip install -r requirements.txt
export ORT_PROVIDERS=CPUExecutionProvider
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## IF CPU-only

### 1. Dockerfile (CPU)

```dockerfile
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 build-essential gcc g++ make \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# requirements.txt should use "onnxruntime" (CPU), not "onnxruntime-gpu"
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY main.py /app/main.py
ENV ORT_PROVIDERS=CPUExecutionProvider FACE_DET_SIZE=480,480
EXPOSE 8000
CMD ["uvicorn","main:app","--host=0.0.0.0","--port=8000","--workers=1"]
```

#### Build & Run (CPU)

```bash
sudo docker build -t face-api-cpu .
sudo docker run --rm -it -p 8000:8000 \
  -e ORT_PROVIDERS=CPUExecutionProvider \
  -v $HOME/.insightface:/root/.insightface \
  face-api-cpu
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)


---