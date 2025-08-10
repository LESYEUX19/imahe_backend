from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import logging
import os
import sqlite3
import datetime
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="IMAHE API", description="AI-powered photo sorting API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "..", "imahe_history.db")
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

user_settings = {"min_exposure": 50.0, "max_exposure": 200.0, "min_sharpness": 100.0, "focused_ratio": 1.75, "eye_aspect_ratio": 8.5}

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, filename TEXT NOT NULL, 
            label TEXT NOT NULL, timestamp TEXT NOT NULL, details_message TEXT, path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at: {DB_FILE}")

@app.on_event("startup")
async def on_startup(): init_db()

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except Exception as e:
    logger.error(f"FATAL: Could not load Haar cascades: {e}")
    face_cascade, eye_cascade = None, None

class ClassificationDetails(BaseModel): message: Optional[str] = None; sharpness: Optional[float] = None; exposure: Optional[float] = None
class ClassificationResult(BaseModel): label: str; details: ClassificationDetails
class SettingsModel(BaseModel): min_exposure: float; max_exposure: float; min_sharpness: float; focused_ratio: float; eye_aspect_ratio: float
class HistoryEntry(BaseModel): id: int; session_id: str; filename: str; label: str; timestamp: str; details_message: Optional[str]; path: str
class LogHistoryRequest(BaseModel): sessionId: str; fileName: str; label: str; detailsMessage: Optional[str]; path: str

@app.get("/settings", response_model=SettingsModel)
async def get_settings(): return user_settings

@app.post("/settings")
async def update_settings(new_settings: SettingsModel):
    user_settings.update(new_settings.dict())
    logger.info(f"Settings updated: {user_settings}")
    return {"message": "Settings updated successfully"}

def calculate_sharpness(image: np.ndarray) -> float:
    if image is None or image.size == 0: return 0.0
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def analyze_faces(image: np.ndarray, ear_threshold: float) -> dict:
    if not face_cascade: return {"count": 0, "has_closed_eyes": False, "face_rois": [], "face_bounds": []}
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    if len(faces) == 0: return {"count": 0, "has_closed_eyes": False, "face_rois": [], "face_bounds": []}
    has_closed_eyes = False; face_rois = []; face_bounds = []
    for (x, y, w, h) in faces:
        if w > 0 and h > 0:
            face_roi_color = image[y:y+h, x:x+w]
            face_rois.append(face_roi_color)
            face_bounds.append((x,y,w,h))
            roi_gray = gray_image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            if len(eyes) > 0:
                is_blinking = all((ew/eh if eh > 0 else 0) > ear_threshold for (ex,ey,ew,eh) in eyes)
                if is_blinking: has_closed_eyes = True
    return {"count": len(faces), "has_closed_eyes": has_closed_eyes, "face_rois": face_rois, "face_bounds": face_bounds}

def classify_image_logic(image_cv2: np.ndarray, settings: dict) -> tuple:
    exposure = float(np.mean(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)))
    if not (settings["min_exposure"] <= exposure <= settings["max_exposure"]): return "Bad", ClassificationDetails(message=f"Poor exposure ({exposure:.1f})", exposure=round(exposure, 2))
    face_analysis = analyze_faces(image_cv2, settings["eye_aspect_ratio"])
    num_faces = face_analysis["count"]
    if face_analysis["has_closed_eyes"]: return "Closed Eye", ClassificationDetails(message="Detected a person with closed eyes.")
    if num_faces > 0:
        face_sharpness_scores = [calculate_sharpness(roi) for roi in face_analysis["face_rois"]]
        if not face_sharpness_scores: return "Good", ClassificationDetails(message="Could not analyze face sharpness.", exposure=round(exposure, 2))
        avg_face_sharpness = np.mean(face_sharpness_scores)
        if avg_face_sharpness < settings["min_sharpness"]: return "Blurred", ClassificationDetails(message=f"Subject(s) are blurry (Avg Sharpness: {avg_face_sharpness:.1f})", sharpness=round(avg_face_sharpness, 2))
        mask = np.ones(image_cv2.shape[:2], dtype="uint8") * 255
        for (x, y, w, h) in face_analysis["face_bounds"]: cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
        background = cv2.bitwise_and(image_cv2, image_cv2, mask=mask)
        background_sharpness = calculate_sharpness(background)
        if background_sharpness > 1 and (avg_face_sharpness / background_sharpness) > settings["focused_ratio"]: return "Focused", ClassificationDetails(message="Subject(s) are in focus.", sharpness=round(avg_face_sharpness, 2))
        return "Good", ClassificationDetails(message=f"Image quality is good (Avg Face Sharpness: {avg_face_sharpness:.1f})", sharpness=round(avg_face_sharpness, 2), exposure=round(exposure, 2))
    else:
        overall_sharpness = calculate_sharpness(image_cv2)
        if overall_sharpness < settings["min_sharpness"]: return "Blurred", ClassificationDetails(message=f"Image is blurry (Sharpness: {overall_sharpness:.1f})", sharpness=round(overall_sharpness, 2))
        return "Good", ClassificationDetails(message=f"Image quality is good (Sharpness: {overall_sharpness:.1f})", sharpness=round(overall_sharpness, 2), exposure=round(exposure, 2))

@app.post("/clear-state/")
async def clear_state():
    for f in os.listdir(IMAGES_DIR): os.remove(os.path.join(IMAGES_DIR, f))
    return {"message": "State cleared"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if not face_cascade: raise HTTPException(status_code=503, detail="Backend not ready: Haar cascades missing.")
    try:
        contents = await file.read()
        temp_path = os.path.join(IMAGES_DIR, file.filename)
        with open(temp_path, "wb") as f: f.write(contents)
        image_cv2 = cv2.cvtColor(np.array(Image.open(io.BytesIO(contents)).convert("RGB")), cv2.COLOR_RGB2BGR)
        label, details = classify_image_logic(image_cv2, user_settings)
        return ClassificationResult(label=label, details=details)
    except Exception as e:
        logger.error(f"Error in upload_image for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")

@app.post("/log-history-entry")
async def log_history_entry(request: LogHistoryRequest):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (session_id, filename, label, timestamp, details_message, path) VALUES (?, ?, ?, ?, ?, ?)",
            (request.sessionId, request.fileName, request.label, datetime.datetime.now().isoformat(), request.detailsMessage, request.path)
        )
        conn.commit()
        conn.close()
        logger.info(f"History entry logged for session {request.sessionId}")
        return {"message": "History logged successfully"}
    except Exception as e:
        # This will now be sent to the frontend if something goes wrong
        logger.error(f"DATABASE ERROR: Failed to log to database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

@app.get("/history", response_model=List[HistoryEntry])
async def get_history():
    if not os.path.exists(DB_FILE): return []
    try:
        conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute("SELECT id, session_id, filename, label, timestamp, details_message, path FROM history ORDER BY timestamp DESC")
        rows = cursor.fetchall(); conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Could not retrieve history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history.")

@app.delete("/history")
async def clear_history():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        init_db()
        logger.info("Processing history has been cleared.")
    return JSONResponse(status_code=200, content={"message": "History cleared successfully."})

@app.get("/history/image")
async def get_history_image(path: str):
    if not os.path.isfile(path):
        logger.error(f"History image not found at path: {path}")
        raise HTTPException(status_code=404, detail=f"Image not found at specified path.")
    return FileResponse(path)