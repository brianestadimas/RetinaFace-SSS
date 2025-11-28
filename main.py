import os
import base64
import asyncio
import json
from typing import List, Dict, Optional, Iterable

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from deepface import DeepFace

SIM_THRESHOLD = 0.75

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

FACE_DB_ROOT = "face_db"
os.makedirs(FACE_DB_ROOT, exist_ok=True)

# live frames storage under face_db/_live
LIVE_DB_ROOT = os.path.join(FACE_DB_ROOT, "_live")
os.makedirs(LIVE_DB_ROOT, exist_ok=True)

deepface_lock = asyncio.Lock()


class EnrollRequest(BaseModel):
    employee_id: str = Field(default="1")
    images: List[str] = Field(..., description="Base64 images, with or without data URL prefix")


class VerifyImageRequest(BaseModel):
    employee_id: str = Field(default="1")
    image: str = Field(..., description="Base64 image, with or without data URL prefix")
    threshold: Optional[float] = Field(default=None, description="Cosine [0,1]")


class ClearRequest(BaseModel):
    employee_id: str = Field(default="1")


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


def _frame_path_for(employee_id: str) -> str:
    safe = (employee_id or "live").replace("/", "_")
    return os.path.join(LIVE_DB_ROOT, f"{safe}.jpg")


async def _decode_image_from_request(request: Request, field_names: Iterable[str] = ("frame", "image")) -> Optional[np.ndarray]:
    ct = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ct:
        form = await request.form()
        for name in field_names:
            file = form.get(name)
            if isinstance(file, UploadFile):
                data = await file.read()
                return _bytes_to_bgr(data)
        for name in field_names:
            raw = form.get(name)
            if isinstance(raw, (bytes, bytearray)):
                return _bytes_to_bgr(bytes(raw))
            if isinstance(raw, str):
                img = _b64_to_bgr(raw)
                if img is not None:
                    return img
        return None
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


def _save_resized(src_path: str, max_side: int = 1280):
    img = cv2.imread(src_path)
    if img is None:
        return src_path, 1.0, 1.0
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return src_path, 1.0, 1.0
    scale = max_side / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_path = src_path + ".small.jpg"
    cv2.imwrite(small_path, small)
    return small_path, (w / float(new_w)), (h / float(new_h))


async def _run_deepface(img_path: str, db_path: str):
    small_path, sx, sy = _save_resized(img_path, max_side=1280)

    extract_kwargs = dict(
        img_path=small_path,
        align=True,
        anti_spoofing=True,
        enforce_detection=True,
    )

    try:
        async with deepface_lock:
            faces = await run_in_threadpool(DeepFace.extract_faces, **extract_kwargs)
    except ValueError:
        return None
    except Exception:
        raise

    if not faces:
        return None

    find_kwargs = dict(
        img_path=img_path,
        db_path=db_path,
        model_name="Buffalo_L",
        distance_metric="cosine",
        enforce_detection=False,
    )

    async with deepface_lock:
        find_res = await run_in_threadpool(DeepFace.find, **find_kwargs)

    best_identity = ""
    best_distance = None
    if isinstance(find_res, list) and len(find_res) > 0 and not find_res[0].empty:
        row = find_res[0].iloc[0]
        best_identity = str(row.get("identity", "")) or ""
        try:
            best_distance = float(row.get("distance"))
        except Exception:
            best_distance = None

    verified = False
    model_threshold = None

    if best_identity:
        verify_kwargs = dict(
            img1_path=img_path,
            img2_path=best_identity,
            model_name="Buffalo_L",
            distance_metric="cosine",
            enforce_detection=False,
            anti_spoofing=False,
        )
        async with deepface_lock:
            vres = await run_in_threadpool(DeepFace.verify, **verify_kwargs)
        verified = bool(vres.get("verified", False))
        model_threshold = vres.get("threshold")
        if best_distance is None:
            best_distance = vres.get("distance")

    f0 = faces[0]
    region = f0.get("facial_area") or f0.get("region") or {}
    x = float(region.get("x", 0)) * sx
    y = float(region.get("y", 0)) * sy
    w = float(region.get("w", 0)) * sx
    h = float(region.get("h", 0)) * sy

    det_score = float(f0.get("confidence", 1.0))
    is_real = bool(f0.get("is_real", True))

    matched_uid = None
    if best_identity:
        parts = best_identity.replace("\\", "/").split("/")
        if len(parts) >= 2:
            matched_uid = parts[-2]

    authorized_flag = verified and is_real

    return {
        "bbox": [x, y, x + w, y + h],
        "match_user": matched_uid,
        "confidence": None if best_distance is None else round(max(0.0, 1.0 - float(best_distance)), 4),
        "authorized": bool(authorized_flag),
        "det_score": det_score,
        "fake": not is_real,
        "model_threshold": model_threshold,
        "raw_distance": best_distance,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/face/enroll", summary="Enroll 2–3 base64 images for an employee_id")
async def face_enroll(payload: EnrollRequest):
    if not payload.images:
        raise HTTPException(status_code=400, detail="No images provided")
    employee_id = payload.employee_id.strip() or "1"
    save_dir = os.path.join(FACE_DB_ROOT, employee_id)
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for s in payload.images:
        img = _b64_to_bgr(s)
        if img is None:
            continue
        out_path = os.path.join(save_dir, f"face_{count}.jpg")
        cv2.imwrite(out_path, img)
        count += 1
    return {"employee_id": employee_id, "added": count, "total": len(os.listdir(save_dir))}


@app.post("/face/enroll-files", summary="Enroll 2–3 uploaded files for an employee_id (multipart)")
async def face_enroll_files(
    employee_id: str = Form(default="1"),
    files: List[UploadFile] = File(default=[]),
):
    employee_id = employee_id.strip() or "1"
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    save_dir = os.path.join(FACE_DB_ROOT, employee_id)
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for f in files:
        try:
            data = await f.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out_path = os.path.join(save_dir, f"face_{count}.jpg")
            cv2.imwrite(out_path, img)
            count += 1
        except Exception:
            continue
    return {"employee_id": employee_id, "added": count, "total": len(os.listdir(save_dir))}


@app.post("/face/verify", summary="Verify one frame via multipart or JSON")
async def face_verify(
    request: Request,
    employee_id: str = Query(default="1"),
    threshold: float = Query(default=SIM_THRESHOLD),
):
    employee_id = employee_id.strip() or "1"
    if employee_id == "*":
        db_path = FACE_DB_ROOT
    else:
        db_path = os.path.join(FACE_DB_ROOT, employee_id)
        if not os.path.isdir(db_path):
            return {
                "employee_id": employee_id,
                "threshold": threshold,
                "detections": [],
                "count": 0,
                "authorized": False,
                "reason": "no stored face",
            }

    img = await _decode_image_from_request(request, ("frame", "image"))
    if img is None:
        raise HTTPException(status_code=400, detail="No image provided")

    frame_path = _frame_path_for(employee_id)
    cv2.imwrite(frame_path, img)

    try:
        det = await _run_deepface(frame_path, db_path)
        if det is None:
            return {
                "employee_id": employee_id,
                "threshold": threshold,
                "detections": [],
                "count": 0,
                "authorized": False,
                "reason": "no face found",
            }
        return {
            "employee_id": employee_id,
            "threshold": threshold,
            "detections": [det],
            "count": 1,
            "authorized": bool(det["authorized"]),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})


@app.websocket("/face/verify")
async def face_verify_ws(
    websocket: WebSocket,
    employee_id: str = Query(default="1"),
    threshold: float = Query(default=SIM_THRESHOLD),
):
    await websocket.accept()
    employee_id = employee_id.strip() or "1"

    try:
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break

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
                try:
                    await websocket.send_json({"error": "no/invalid frame"})
                except (RuntimeError, WebSocketDisconnect):
                    break
                continue

            if employee_id == "*":
                db_path = FACE_DB_ROOT
            else:
                db_path = os.path.join(FACE_DB_ROOT, employee_id)
                if not os.path.isdir(db_path):
                    payload = {
                        "employee_id": employee_id,
                        "threshold": threshold,
                        "detections": [],
                        "count": 0,
                        "authorized": False,
                        "reason": "no stored face",
                    }
                    try:
                        await websocket.send_json(payload)
                    except (RuntimeError, WebSocketDisconnect):
                        break
                    continue

            frame_path = _frame_path_for(employee_id)
            cv2.imwrite(frame_path, img)

            try:
                det = await _run_deepface(frame_path, db_path)
            except Exception as e:
                try:
                    await websocket.send_json({"error": f"{type(e).__name__}: {e}"})
                except (RuntimeError, WebSocketDisconnect):
                    break
                continue

            if det is None:
                payload = {
                    "employee_id": employee_id,
                    "threshold": threshold,
                    "detections": [],
                    "count": 0,
                    "authorized": False,
                    "reason": "no face found",
                }
            else:
                payload = {
                    "employee_id": employee_id,
                    "threshold": threshold,
                    "detections": [det],
                    "count": 1,
                    "authorized": bool(det["authorized"]),
                }

            try:
                await websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                break
    finally:
        # nothing special, connection is already closed by client or server
        return


@app.post("/face/verify-image", summary="Verify a single captured image (JSON, data URL/base64)")
async def face_verify_image(payload: VerifyImageRequest):
    img = _b64_to_bgr(payload.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or missing image")
    employee_id = payload.employee_id.strip() or "1"
    thr = payload.threshold if payload.threshold is not None else SIM_THRESHOLD

    if employee_id == "*":
        db_path = FACE_DB_ROOT
    else:
        db_path = os.path.join(FACE_DB_ROOT, employee_id)
        if not os.path.isdir(db_path):
            return {
                "employee_id": employee_id,
                "threshold": thr,
                "detections": [],
                "count": 0,
                "authorized": False,
                "reason": "no stored face",
            }

    frame_path = _frame_path_for(employee_id)
    cv2.imwrite(frame_path, img)

    det = await _run_deepface(frame_path, db_path)
    if det is None:
        return {
            "employee_id": employee_id,
            "threshold": thr,
            "detections": [],
            "count": 0,
            "authorized": False,
            "reason": "no face found",
        }
    return {
        "employee_id": employee_id,
        "threshold": thr,
        "detections": [det],
        "count": 1,
        "authorized": bool(det["authorized"]),
    }


@app.post("/face/clear", summary="Clear embeddings for an employee_id")
async def face_clear(payload: ClearRequest):
    employee_id = payload.employee_id.strip() or "1"
    folder = os.path.join(FACE_DB_ROOT, employee_id)
    removed = 0
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
                removed += 1
            except Exception:
                pass
        try:
            os.rmdir(folder)
        except Exception:
            pass
    return {"employee_id": employee_id, "removed": removed}
