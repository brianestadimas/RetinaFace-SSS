import os
import base64
import json
import tempfile
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, Request, Query, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from deepface import DeepFace
import asyncio
from starlette.concurrency import run_in_threadpool

deepface_lock = asyncio.Lock()

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
app = FastAPI(title="Face Verify API (DeepFace GPU)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

FACE_DB_ROOT = "face_db"
os.makedirs(FACE_DB_ROOT, exist_ok=True)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def decode_image_from_request(req: Request, field="frame"):
    """Decode multipart or JSON base64 to OpenCV BGR."""
    # multipart
    if req.headers.get("content-type", "").startswith("multipart"):
        form = req.form()
        return_form = req.scope.get("_form")
        return None

    # Handle multipart through FastAPI UploadFile
    # This path is cleaner:
    # Not reading req directly, but from uploaded files:
    return None


async def decode_image(request: Request, field="frame"):
    ct = request.headers.get("content-type", "").lower()

    # multipart
    if "multipart/form-data" in ct:
        form = await request.form()
        file = form.get(field)
        if isinstance(file, UploadFile):
            data = await file.read()
            arr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # JSON
    if "application/json" in ct:
        obj = await request.json()
        data_url = obj.get(field)
        if isinstance(data_url, str) and data_url.startswith("data:"):
            _, b64 = data_url.split(",", 1)
            data = base64.b64decode(b64)
            arr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return None


def save_resized(path, max_side=1280):
    """Downscale for faster detection; return new path + inverse scale."""
    img = cv2.imread(path)
    if img is None:
        return path, 1.0, 1.0

    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return path, 1.0, 1.0

    s = max_side / float(max(h, w))
    nw, nh = int(w * s), int(h * s)

    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    small_path = path + ".small.jpg"
    cv2.imwrite(small_path, small)

    return small_path, (w / float(nw)), (h / float(nh))


# ----------------------------------------------------------
# Health
# ----------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------------------------------------
# ENROLL — save files to disk
# ----------------------------------------------------------
@app.post("/face/enroll")
async def face_enroll(
    user_id: str = Form(default="authorized"),
    files: list[UploadFile] = File(default=[]),
):
    user_id = user_id.strip() or "authorized"

    if not files:
        raise HTTPException(400, "No files uploaded")

    save_dir = os.path.join(FACE_DB_ROOT, user_id)
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
            pass

    return {
        "user_id": user_id,
        "added": count,
        "total": len(os.listdir(save_dir)),
    }


# ----------------------------------------------------------
# VERIFY — DeepFace anti-spoof + recognition
# ----------------------------------------------------------
@app.post("/face/verify")
async def face_verify(
    request: Request,
    user_id: str = Query(default="authorized"),
    detector: str = Query(default=None),
):
    user_id = user_id.strip() or "authorized"
    folder = os.path.join(FACE_DB_ROOT, user_id)

    if not os.path.isdir(folder):
        return {
            "user_id": user_id,
            "authorized": False,
            "reason": "no stored face",
        }

    img = await decode_image(request, "frame")
    if img is None:
        raise HTTPException(400, "No image provided")

    # save temp original
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp.name, img)

    try:
        # fast detect on downscaled version
        small_path, sx, sy = save_resized(tmp.name, max_side=1280)

        extract_kwargs = dict(
            img_path=small_path,
            align=True,
            anti_spoofing=True,
            enforce_detection=True,
        )
        if detector:
            extract_kwargs["detector_backend"] = detector

        async with deepface_lock:
            faces = await run_in_threadpool(DeepFace.extract_faces, **extract_kwargs)


        if not faces:
            return {
                "user_id": user_id,
                "authorized": False,
                "reason": "no face found",
            }

        # gallery search
        find_kwargs = dict(
            img_path=tmp.name,
            db_path=folder,
            model_name="Buffalo_L",
            distance_metric="cosine",
            enforce_detection=False,
        )
        if detector:
            find_kwargs["detector_backend"] = detector

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

        # verify()
        verified = False
        model_threshold = None

        if best_identity:
            verify_kwargs = dict(
                img1_path=tmp.name,
                img2_path=best_identity,
                model_name="Buffalo_L",
                distance_metric="cosine",
                enforce_detection=False,
                anti_spoofing=False,
            )
            if detector:
                verify_kwargs["detector_backend"] = detector

            try:
                res = DeepFace.verify(**verify_kwargs)
                verified = bool(res.get("verified", False))
                model_threshold = res.get("threshold")
                if best_distance is None:
                    best_distance = res.get("distance")
            except Exception:
                verified = False

        # bbox rescale
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

        detections = [{
            "bbox": [x, y, x + w, y + h],
            "match_user": matched_uid,
            "confidence": None if best_distance is None else round(max(0.0, 1.0 - float(best_distance)), 4),
            "authorized": bool(authorized_flag),
            "det_score": det_score,
            "fake": not is_real,
            "model_threshold": model_threshold,
            "raw_distance": best_distance,
        }]

        return {
            "user_id": user_id,
            "authorized": bool(authorized_flag),
            "detector_backend": detector,
            "detections": detections,
            "count": len(detections),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e).__name__}: {e}"}
        )

    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass
        try:
            os.unlink(tmp.name + ".small.jpg")
        except:
            pass


# ----------------------------------------------------------
# CLEAR
# ----------------------------------------------------------
@app.post("/face/clear")
async def face_clear(user_id: str = Form(default="authorized")):
    user_id = user_id.strip() or "authorized"
    folder = os.path.join(FACE_DB_ROOT, user_id)

    if os.path.isdir(folder):
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except:
                pass
        try:
            os.rmdir(folder)
        except:
            pass
        return {"user_id": user_id, "cleared": True}

    return {"user_id": user_id, "cleared": False}
