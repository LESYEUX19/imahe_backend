from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import cv2
import numpy as np
from PIL import Image
import io, logging, os, sqlite3, datetime, re
from typing import List, Optional, Dict
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import imagehash

# --- Setup ---
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
user_settings = {"min_exposure": 50.0, "max_exposure": 200.0, "min_sharpness": 100.0, "focused_ratio": 1.75, "eye_aspect_ratio": 4.0}
session_hashes: Dict[imagehash.ImageHash, str] = {}

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.cursor().execute("""
            CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, filename TEXT NOT NULL, label TEXT NOT NULL, timestamp TEXT NOT NULL, details_message TEXT, path TEXT NOT NULL)
        """)
    logger.info("Database initialized.")

@app.on_event("startup")
async def on_startup():
    init_db()

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except Exception as e:
    logger.error(f"FATAL: Could not load Haar cascades: {e}")
    face_cascade, eye_cascade = None, None

# --- Pydantic Models ---
class ClassificationDetails(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    message: Optional[str] = None; exposure: Optional[float] = None; sharpness: Optional[float] = None
    face_count: Optional[int] = Field(default=None, alias="faceCount")
    is_duplicate: bool = Field(default=False, alias="isDuplicate")
    has_closed_eyes: bool = Field(default=False, alias="hasClosedEyes")
class ClassificationResult(BaseModel): label: str; details: ClassificationDetails
class UploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    classification: ClassificationResult
    sanitized_filename: str = Field(..., alias="sanitizedFilename")
    image_url: str = Field(..., alias="imageUrl")
class SettingsModel(BaseModel): min_exposure: float; max_exposure: float; min_sharpness: float; focused_ratio: float; eye_aspect_ratio: float
class HistoryEntry(BaseModel): id: int; session_id: str; filename: str; label: str; timestamp: str; details_message: Optional[str]; path: str
class LogHistoryRequest(BaseModel): sessionId: str; fileName: str; label: str; detailsMessage: Optional[str]; path: str

# --- Helper Functions ---
def sanitize_filename(filename: str) -> str: return re.sub(r'[^a-zA-Z0-9._-]', '', os.path.basename(filename))
def calculate_sharpness(image: np.ndarray) -> float:
    if image is None or image.size == 0: return 0.0
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
def analyze_faces(image: np.ndarray, ear_threshold: float) -> dict:
    if not face_cascade: return {"count": 0, "has_closed_eyes": False, "face_rois": [], "face_bounds": []}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if not len(faces): return {"count": 0, "has_closed_eyes": False, "face_rois": [], "face_bounds": []}
    face_rois, face_bounds = [], []
    for (x, y, w, h) in faces:
        if w > 0 and h > 0:
            face_rois.append(image[y:y+h, x:x+w]); face_bounds.append((x, y, w, h))
            roi_gray = gray[y:y+h, x:x+w]; eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            if len(eyes) < 2: return {"count": len(faces), "has_closed_eyes": True, "face_rois": face_rois, "face_bounds": face_bounds}
            for (ex, ey, ew, eh) in eyes:
                if eh > 0 and (ew / eh) > ear_threshold: return {"count": len(faces), "has_closed_eyes": True, "face_rois": face_rois, "face_bounds": face_bounds}
    return {"count": len(faces), "has_closed_eyes": False, "face_rois": face_rois, "face_bounds": face_bounds}
def is_line_art_or_graphic(gray_image: np.ndarray, threshold=0.95) -> bool:
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    total_pixels = gray_image.size
    return total_pixels > 0 and (hist[0] + hist[255]) / total_pixels > threshold

def classify_image_logic(image_pil: Image.Image, image_cv2: np.ndarray, original_filename: str, settings: dict) -> tuple:
    try:
        hash_val = imagehash.phash(image_pil)
        if hash_val in session_hashes:
            duplicate_of = session_hashes[hash_val]
            message = f"Duplicate of: {duplicate_of}"
            return "Duplicate", ClassificationDetails(is_duplicate=True, message=message)
        session_hashes[hash_val] = original_filename
    except Exception as e:
        logger.warning(f"Could not calculate image hash for {original_filename}: {e}")
    
    gray_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    exposure = float(np.mean(gray_image))
    face_analysis = analyze_faces(image_cv2, settings["eye_aspect_ratio"])
    face_count = face_analysis["count"]
    if face_analysis["has_closed_eyes"]: return "Closed Eye", ClassificationDetails(has_closed_eyes=True, face_count=face_count, exposure=round(exposure, 2), message="A person may have closed eyes.")
    if face_count > 0:
        face_sharpness_scores = [calculate_sharpness(roi) for roi in face_analysis["face_rois"]]
        if not face_sharpness_scores: return "Good", ClassificationDetails(face_count=face_count, exposure=round(exposure, 2), message="Face detected but sharpness could not be analyzed.")
        avg_face_sharpness = np.mean(face_sharpness_scores)
        if avg_face_sharpness < settings["min_sharpness"]: return "Blurred", ClassificationDetails(sharpness=round(avg_face_sharpness, 2), face_count=face_count, exposure=round(exposure, 2), message="Subject appears to be blurry.")
        mask = np.ones(image_cv2.shape[:2], dtype="uint8") * 255
        for (x, y, w, h) in face_analysis["face_bounds"]: cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
        background_sharpness = calculate_sharpness(cv2.bitwise_and(image_cv2, image_cv2, mask=mask))
        if background_sharpness > 10.0 and (avg_face_sharpness / background_sharpness) > settings["focused_ratio"]: return "Focused", ClassificationDetails(sharpness=round(avg_face_sharpness, 2), face_count=face_count, exposure=round(exposure, 2), message="Subject is sharp and in focus.")
        return "Good", ClassificationDetails(sharpness=round(avg_face_sharpness, 2), face_count=face_count, exposure=round(exposure, 2), message="Image is good.")
    else:
        if is_line_art_or_graphic(gray_image): return "Good", ClassificationDetails(sharpness=0, face_count=0, exposure=round(exposure, 2), message="Image detected as a graphic or line art.")
        overall_sharpness = calculate_sharpness(image_cv2)
        if overall_sharpness < settings["min_sharpness"]: return "Blurred", ClassificationDetails(sharpness=round(overall_sharpness, 2), face_count=0, exposure=round(exposure, 2), message="The image appears to be blurry.")
        return "Good", ClassificationDetails(sharpness=round(overall_sharpness, 2), face_count=0, exposure=round(exposure, 2), message="Image is good, no faces detected.")

@app.post("/upload-image/", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    if not face_cascade: raise HTTPException(status_code=503, detail="Backend not ready")
    original_filename = file.filename
    safe_filename = sanitize_filename(original_filename)
    if not safe_filename: raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        label, details = classify_image_logic(image_pil, image_cv2, original_filename, user_settings)
        save_path = os.path.join(IMAGES_DIR, safe_filename)
        with open(save_path, "wb") as f: f.write(contents)
        web_path = os.path.join("static", "images", safe_filename).replace("\\", "/")
        return UploadResponse(classification=ClassificationResult(label=label, details=details), sanitized_filename=safe_filename, image_url=web_path)
    except Exception as e:
        logger.error(f"Error in upload_image for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")

@app.post("/clear-state/")
async def clear_state():
    session_hashes.clear()
    for f in os.listdir(IMAGES_DIR):
        os.remove(os.path.join(IMAGES_DIR, f))
    return {"message": "State cleared"}

@app.get("/settings", response_model=SettingsModel)
async def get_settings():
    return user_settings

@app.post("/settings")
async def update_settings(new_settings: SettingsModel):
    user_settings.update(new_settings.dict())
    return {"message": "Settings updated"}

@app.post("/log-history-entry")
async def log_history_entry(request: LogHistoryRequest):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("INSERT INTO history (session_id, filename, label, timestamp, details_message, path) VALUES (?, ?, ?, ?, ?, ?)", (request.sessionId, request.fileName, request.label, datetime.datetime.now().isoformat(), request.detailsMessage, request.path))
            conn.commit()
        return {"message": "History logged successfully"}
    except Exception as e:
        logger.error(f"DATABASE ERROR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

@app.get("/history", response_model=List[HistoryEntry])
async def get_history():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, session_id, filename, label, timestamp, details_message, path FROM history ORDER BY timestamp DESC").fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Could not retrieve history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve history.")

@app.delete("/history")
async def clear_history():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM history")
            conn.commit()
        init_db()
        return JSONResponse(status_code=200, content={"message": "History cleared successfully."})
    except Exception as e:
        logger.error(f"Failed to clear history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear history.")