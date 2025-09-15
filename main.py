import os
import base64
import asyncio
import json
from typing import List, Dict, Optional, Iterable

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel, Field
from insightface.app import FaceAnalysis
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

# -----------------------
# Config & Globals
# -----------------------
SIM_THRESHOLD = 0.75
DET_SIZE = tuple(map(int, os.environ.get("FACE_DET_SIZE", "480,480").split(",")))
PROVIDERS = os.environ.get("ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",")

app = FastAPI(
    title="Face Verification API",
    version="1.0.2",
    description=(
        "Endpoints:\n"
        "- POST /face/enroll: enroll 2–3 base64 images for an employee_id\n"
        "- POST /face/enroll-files: enroll 2–3 uploaded files for an employee_id\n"
        "- POST /face/verify: verify one frame (multipart file or JSON base64)\n"
        "- WEBSOCKET /face/verify: stream frames (binary or JSON base64)\n"
        "- POST /face/verify-image: verify a single captured image (JSON)\n"
        "- POST /face/clear: clear embeddings for an employee_id\n"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_face_app: Optional[FaceAnalysis] = None
_face_lock = asyncio.Lock()
_db_lock = asyncio.Lock()
FACE_DB: Dict[str, List[List[float]]] = {}  # employee_id -> list[embedding]


# -----------------------
# Models
# -----------------------
class EnrollRequest(BaseModel):
    employee_id: str = Field(default="1")
    images: List[str] = Field(..., description="Base64 images, with or without data URL prefix")

class VerifyImageRequest(BaseModel):
    employee_id: str = Field(default="1")
    image: str = Field(..., description="Base64 image, with or without data URL prefix")
    threshold: Optional[float] = Field(default=None, description="Cosine [0,1]")

class ClearRequest(BaseModel):
    employee_id: str = Field(default="1")


# -----------------------
# Utils
# -----------------------
def _strip_b64(s: str) -> str:
    if isinstance(s, str) and "," in s and s.lstrip().lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s

def _b64_to_bgr(b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(_strip_b64(b64), validate=False)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _bytes_to_bgr(data: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _to_nd(v):
    if not v:
        return np.zeros((0, 512), dtype=np.float32)
    a = np.asarray(v, dtype=np.float32)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    return a

def _proto_for(user_id: str) -> np.ndarray:
    vecs = _to_nd(FACE_DB.get(user_id, []))
    if vecs.shape[0] == 0:
        return np.zeros((512,), dtype=np.float32)
    p = vecs.mean(axis=0)
    p /= (np.linalg.norm(p) + 1e-9)
    return p.astype(np.float32)

async def _extract_best_face_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    async with _face_lock:
        faces = await run_in_threadpool(_face_app.get, img_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return face.normed_embedding.astype(np.float32)

async def _verify_embeddings(img_bgr: np.ndarray, employee_id: str, threshold: float):
    async with _face_lock:
        faces = await run_in_threadpool(_face_app.get, img_bgr)

    if employee_id == "*":
        protos = {}
        async with _db_lock:
            for uid in FACE_DB.keys():
                protos[uid] = _proto_for(uid)
    else:
        protos = {employee_id: _proto_for(employee_id)}

    out = []
    for f in faces:
        emb = f.normed_embedding.astype(np.float32)
        best_uid, best_sim = None, -1.0
        for uid, proto in protos.items():
            if proto.sum() == 0:
                continue
            sim = float(np.dot(emb, proto))
            if sim > best_sim:
                best_sim, best_uid = sim, uid
        confidence = (best_sim + 1.0) / 2.0
        authorized = bool(best_sim >= (2 * threshold - 1.0))
        out.append({
            "bbox": [float(f.bbox[0]), float(f.bbox[1]), float(f.bbox[2]), float(f.bbox[3])],
            "match_user": best_uid,
            "cosine_sim": round(best_sim, 4),
            "confidence": round(confidence, 4),
            "authorized": authorized,
            "det_score": float(f.det_score),
        })
    return out

async def _enroll_ndarrays(employee_id: str, imgs_bgr: List[np.ndarray]) -> dict:
    added = 0
    async with _db_lock:
        stored = FACE_DB.get(employee_id, [])
    for img in imgs_bgr:
        if img is None:
            continue
        emb = await _extract_best_face_embedding(img)
        if emb is None:
            continue
        stored.append(emb.tolist())
        added += 1
    async with _db_lock:
        FACE_DB[employee_id] = stored
    return {"employee_id": employee_id, "added": added, "total": len(stored)}

async def _decode_image_from_request(request: Request, field_names: Iterable[str] = ("frame", "image")) -> Optional[np.ndarray]:
    ct = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ct:
        form = await request.form()
        # accept UploadFile in any of the allowed fields
        for name in field_names:
            file = form.get(name)
            if isinstance(file, UploadFile):
                data = await file.read()
                return _bytes_to_bgr(data)
        # also accept raw bytes pasted as form field
        for name in field_names:
            raw = form.get(name)
            if isinstance(raw, (bytes, bytearray)):
                return _bytes_to_bgr(bytes(raw))
            if isinstance(raw, str):  # allow data URL/base64 in form
                img = _b64_to_bgr(raw)
                if img is not None:
                    return img
        return None
    # JSON with {"frame": "data:image/..."} or {"image": "base64..."}
    if "application/json" in ct or "text/json" in ct:
        try:
            obj = await request.json()
        except Exception:
            return None
        for name in field_names:
            val = obj.get(name)
            if isinstance(val, str):
                img = _b64_to_bgr(val)
                if img is not None:
                    return img
        return None
    return None


# -----------------------
# Lifespan
# -----------------------
@app.on_event("startup")
async def _startup():
    global _face_app
    _face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    _face_app.prepare(ctx_id=0, det_size=DET_SIZE)


# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}  # Health check

@app.post("/face/enroll", summary="Enroll 2–3 base64 images for an employee_id")
async def face_enroll(payload: EnrollRequest):
    if not payload.images:
        raise HTTPException(status_code=400, detail="No images provided")
    imgs = []
    for s in payload.images:
        img = _b64_to_bgr(s)
        if img is not None:
            imgs.append(img)
    return await _enroll_ndarrays(payload.employee_id, imgs)  # JSON enroll (base64)

@app.post("/face/enroll-files", summary="Enroll 2–3 uploaded files for an employee_id (multipart)")
async def face_enroll_files(
    employee_id: str = Form(default="1"),
    files: List[UploadFile] = File(default=[]),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    imgs = []
    for f in files:
        data = await f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    return await _enroll_ndarrays(employee_id, imgs)  # Multipart enroll (files)

@app.post("/face/verify", summary="Verify one frame via multipart or JSON")
async def face_verify(request: Request, employee_id: str = Query(default="1"), threshold: float = Query(default=SIM_THRESHOLD)):
    content_type = request.headers.get("content-type", "")
    img = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("frame") or form.get("image")
        if file and hasattr(file, "read"):
            data = await file.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    elif "application/json" in content_type:
        obj = await request.json()
        frame = obj.get("frame") or obj.get("image")
        if isinstance(frame, str):
            if frame.startswith("data:"):
                _, b64 = frame.split(",", 1)
                data = base64.b64decode(b64)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="No image provided")

    detections = await _verify_embeddings(img, employee_id, threshold)
    return {
        "employee_id": employee_id,
        "threshold": threshold,
        "detections": detections,
        "count": len(detections),
    }  # Verify (file or base64)

@app.websocket("/face/verify")
async def face_verify_ws(websocket: WebSocket, employee_id: str = Query(default="1"), threshold: float = Query(default=SIM_THRESHOLD)):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            img = None
            if "bytes" in msg and msg["bytes"] is not None:
                img = _bytes_to_bgr(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                try:
                    obj = json.loads(msg["text"])
                    s = obj.get("frame") or obj.get("image")
                    if isinstance(s, str):
                        img = _b64_to_bgr(s)
                except Exception:
                    img = None
            if img is None:
                await websocket.send_json({"error": "no/invalid frame"})
                continue
            detections = await _verify_embeddings(img, employee_id, threshold)
            await websocket.send_json({
                "employee_id": employee_id,
                "threshold": threshold,
                "detections": detections,
                "count": len(detections),
            })
    except WebSocketDisconnect:
        return  # Stream verify (WebSocket)

@app.post("/face/verify-image", summary="Verify a single captured image (JSON, data URL/base64)")
async def face_verify_image(payload: VerifyImageRequest):
    img = _b64_to_bgr(payload.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or missing image")
    thr = payload.threshold if payload.threshold is not None else SIM_THRESHOLD
    detections = await _verify_embeddings(img, payload.employee_id, thr)
    return {"employee_id": payload.employee_id, "threshold": thr, "detections": detections, "count": len(detections)}  # Single image verify

@app.post("/face/clear", summary="Clear embeddings for an employee_id")
async def face_clear(payload: ClearRequest):
    async with _db_lock:
        removed = len(FACE_DB.get(payload.employee_id, []))
        FACE_DB.pop(payload.employee_id, None)
    return {"employee_id": payload.employee_id, "removed": removed}  # Clear database
