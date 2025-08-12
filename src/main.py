# main.py - FULL CORRECTED CODE
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io, logging, os, sqlite3, datetime, re, uuid, json
from typing import List, Optional, Dict
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import imagehash
import dlib
from scipy.spatial import distance as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="IMAHE API", description="AI-powered photo sorting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "..", "imahe_history.db")
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")
HISTORY_IMAGES_DIR = os.path.join(STATIC_DIR, "history_images")
os.makedirs(HISTORY_IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DLIB_LANDMARK_PREDICTOR_PATH = os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat")
PROTOTXT_PATH = os.path.join(SCRIPT_DIR, "deploy.prototxt.txt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

dlib_landmark_predictor = None
dnn_face_detector = None
if not all(os.path.exists(p) for p in [DLIB_LANDMARK_PREDICTOR_PATH, PROTOTXT_PATH, MODEL_PATH]):
    logger.error("FATAL: One or more AI model files are missing.")
else:
    dlib_landmark_predictor = dlib.shape_predictor(DLIB_LANDMARK_PREDICTOR_PATH)
    dnn_face_detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    logger.info("AI Models loaded successfully at startup.")

SETTINGS_FILE = os.path.join(SCRIPT_DIR, "..", "settings.json")
DEFAULT_SETTINGS = {
    "min_exposure": 50.0,
    "max_exposure": 200.0,
    "min_sharpness": 100.0,
    "focused_ratio": 1.75,
    "very_sharp_threshold": 250.0,
    "eye_aspect_ratio": 0.25,
    "closed_eye_percentage": 0.5,
    "duplicate_hash_distance": 5
}

def save_settings_to_file(settings_dict):
    with open(SETTINGS_FILE, 'w') as f: json.dump(settings_dict, f, indent=4)
    logger.info(f"Settings saved to {SETTINGS_FILE}")

def load_settings_from_file():
    if not os.path.exists(SETTINGS_FILE):
        save_settings_to_file(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            for key, value in DEFAULT_SETTINGS.items(): settings.setdefault(key, value)
            return settings
    except (json.JSONDecodeError, IOError):
        logger.error(f"Could not read {SETTINGS_FILE}, falling back to defaults."); save_settings_to_file(DEFAULT_SETTINGS); return DEFAULT_SETTINGS

user_settings = load_settings_from_file()
session_hashes: Dict[imagehash.ImageHash, str] = {}

class ClassificationDetails(BaseModel): model_config = ConfigDict(populate_by_name=True); message: Optional[str] = None; exposure: Optional[float] = None; sharpness: Optional[float] = None; face_count: Optional[int] = Field(default=None, alias="faceCount"); is_duplicate: bool = Field(default=False, alias="isDuplicate"); has_closed_eyes: bool = Field(default=False, alias="hasClosedEyes")
class ClassificationResult(BaseModel): label: str; details: ClassificationDetails
class UploadResponse(BaseModel): model_config = ConfigDict(populate_by_name=True); classification: ClassificationResult; sanitized_filename: str = Field(..., alias="sanitizedFilename"); image_url: str = Field(..., alias="imageUrl")
class SettingsModel(BaseModel):
    min_exposure: float
    max_exposure: float
    min_sharpness: float
class UpdateHistoryLabelRequest(BaseModel): path: str; newLabel: str; newDetailsMessage: str

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, filename TEXT NOT NULL, label TEXT NOT NULL, timestamp TEXT NOT NULL, details_message TEXT, path TEXT NOT NULL UNIQUE, sharpness REAL, exposure REAL)")
    logger.info("Database initialized.")

@app.on_event("startup")
async def on_startup():
    global user_settings
    user_settings = load_settings_from_file()
    init_db()

def eye_aspect_ratio(eye): A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3]); return (A + B) / (2.0 * C)
def analyze_faces_hybrid(image: np.ndarray, ear_threshold: float, confidence_threshold=0.5):
    if not dnn_face_detector or not dlib_landmark_predictor: return {"count": 0, "closed_eye_count": 0, "face_rois": []}
    (h, w) = image.shape[:2]
    if h == 0 or w == 0: return {"count": 0, "closed_eye_count": 0, "face_rois": []}
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)); dnn_face_detector.setInput(blob); detections = dnn_face_detector.forward()
    face_count = 0; closed_eye_count = 0; face_rois = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            face_count += 1; box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]); (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY)); (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            if startX >= endX or startY >= endY: continue
            face_rois.append(image[startY:endY, startX:endX]); rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); shape = dlib_landmark_predictor(gray, rect); shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            leftEye = shape[42:48]; rightEye = shape[36:42]; leftEAR = eye_aspect_ratio(leftEye); rightEAR = eye_aspect_ratio(rightEye); ear = (leftEAR + rightEAR) / 2.0
            if ear < ear_threshold: closed_eye_count += 1
    return {"count": face_count, "closed_eye_count": closed_eye_count, "face_rois": face_rois}

def sanitize_filename(filename: str) -> str: return re.sub(r'[^a-zA-Z0-9._-]', '', os.path.basename(filename))
def calculate_sharpness(image: np.ndarray) -> float:
    if image is None or image.size == 0: return 0.0
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
def is_line_art_or_graphic(gray_image: np.ndarray, threshold=0.95) -> bool: hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]); total_pixels = gray_image.size; return total_pixels > 0 and (hist[0] + hist[255]) / total_pixels > threshold
def _resize_for_analysis(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    try: h, w = image.shape[:2]; _= max(h, w); scale = max_dim / _; new_w, new_h = int(w * scale), int(h * scale); return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception: return image

def classify_image_logic(image_pil: Image.Image, image_cv2: np.ndarray, original_filename: str, settings: dict) -> tuple:
    try:
        phash = imagehash.phash(image_pil)
        if phash in session_hashes:
            return "Duplicate", ClassificationDetails(is_duplicate=True, message=f"Duplicate of: {session_hashes[phash]}")
        session_hashes[phash] = original_filename
    except Exception as e:
        logger.warning(f"Could not calculate image hash for {original_filename}: {e}")
    image_cv2_small = _resize_for_analysis(image_cv2); gray_image_small = cv2.cvtColor(image_cv2_small, cv2.COLOR_BGR2GRAY); exposure = float(np.mean(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY))); overall_sharpness = calculate_sharpness(image_cv2_small)
    
    # --- THIS IS THE CRITICAL CHANGE ---
    if exposure < settings['min_exposure'] or exposure > settings['max_exposure']:
        return "Poor Quality", ClassificationDetails(exposure=round(exposure, 2), message="Image is too dark or too bright.")
        
    face_analysis = analyze_faces_hybrid(image_cv2_small, settings["eye_aspect_ratio"]); face_count = face_analysis["count"]
    if face_count > 0 and face_analysis["closed_eye_count"] / face_count >= settings["closed_eye_percentage"]:
        msg = "Eyes Closed Detected: Please Review"
        return "Closed Eye", ClassificationDetails(has_closed_eyes=True, face_count=face_count, exposure=round(exposure, 2), message=msg)
    if face_count > 0:
        face_sharpness_scores = [calculate_sharpness(roi) for roi in face_analysis["face_rois"]]
        if not face_sharpness_scores: return "Good", ClassificationDetails(face_count=face_count, exposure=round(exposure, 2), message="Face detected but sharpness could not be analyzed.")
        max_face_sharpness = max(face_sharpness_scores); is_very_sharp = max_face_sharpness > settings["very_sharp_threshold"]; has_high_ratio = overall_sharpness > 0 and (max_face_sharpness / overall_sharpness) > settings["focused_ratio"]
        if is_very_sharp and has_high_ratio: return "Focused", ClassificationDetails(sharpness=round(max_face_sharpness, 2), face_count=face_count, exposure=round(exposure, 2), message="Subject is well-focused against the background.")
        if max_face_sharpness < settings["min_sharpness"]: return "Blurred", ClassificationDetails(sharpness=round(max_face_sharpness, 2), face_count=face_count, exposure=round(exposure, 2), message="The main subject appears to be blurry.")
        return "Good", ClassificationDetails(sharpness=round(np.mean(face_sharpness_scores), 2), face_count=face_count, exposure=round(exposure, 2), message="Image is good.")
    else:
        if is_line_art_or_graphic(gray_image_small): return "Good", ClassificationDetails(sharpness=0, face_count=0, exposure=round(exposure, 2), message="Image detected as a graphic or line art.")
        if overall_sharpness < settings["min_sharpness"]: return "Blurred", ClassificationDetails(sharpness=round(overall_sharpness, 2), face_count=0, exposure=round(exposure, 2), message="The image appears to be blurry.")
        return "Good", ClassificationDetails(sharpness=round(overall_sharpness, 2), face_count=0, exposure=round(exposure, 2), message="Image is good.")

@app.post("/upload-image/", response_model=UploadResponse)
async def upload_image(session_id: str = Form(...), file: UploadFile = File(...)):
    original_filename = file.filename; safe_filename = sanitize_filename(original_filename)
    contents = await file.read()
    try: image_pil = Image.open(io.BytesIO(contents)).convert("RGB"); image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        label = "Unreadable"; details = ClassificationDetails(message=f"File '{original_filename}' is corrupted.")
        return UploadResponse(classification=ClassificationResult(label=label, details=details), sanitized_filename=safe_filename, image_url="")
    try:
        label, details = classify_image_logic(image_pil, image_cv2, original_filename, user_settings)
        if label == "Duplicate":
            return UploadResponse(classification=ClassificationResult(label=label, details=details), sanitized_filename=original_filename, image_url="")
        file_extension = os.path.splitext(safe_filename)[1]; unique_history_filename = f"{uuid.uuid4()}{file_extension}"
        permanent_disk_path = os.path.join(HISTORY_IMAGES_DIR, unique_history_filename); permanent_web_path = os.path.join("static", "history_images", unique_history_filename).replace("\\", "/")
        with open(permanent_disk_path, "wb") as f: f.write(contents)
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("INSERT INTO history (session_id, filename, label, timestamp, details_message, path, sharpness, exposure) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (session_id, original_filename, label, datetime.datetime.now().isoformat(), details.message, permanent_web_path, details.sharpness, details.exposure))
            conn.commit()
        return UploadResponse(classification=ClassificationResult(label=label, details=details), sanitized_filename=safe_filename, image_url=permanent_web_path)
    except Exception as e:
        logger.error(f"Error in CLASSIFICATION logic for {file.filename}: {e}", exc_info=True)
        label = "Error"; details = ClassificationDetails(message=f"Backend analysis failed. See console for details.")
        return UploadResponse(classification=ClassificationResult(label=label, details=details), sanitized_filename=safe_filename, image_url="")

@app.post("/clear-state/")
async def clear_state():
    logger.info("Clearing session-specific duplicate cache.")
    session_hashes.clear()
    return {"message": "Session duplicate cache cleared successfully"}

@app.get("/settings", response_model=SettingsModel)
async def get_settings():
    return user_settings

@app.post("/settings")
async def update_settings(new_settings: SettingsModel):
    global user_settings
    update_data = new_settings.dict(exclude_unset=True)
    user_settings.update(update_data)
    save_settings_to_file(user_settings)
    return {"message": "Settings updated successfully"}

@app.put("/update-history-label/")
async def update_history_label(request: UpdateHistoryLabelRequest):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE history SET label = ?, details_message = ? WHERE path = ?", (request.newLabel, request.newDetailsMessage, request.path))
            conn.commit()
            if cursor.rowcount == 0: logger.warning(f"No history entry found to update for path {request.path}")
            else: logger.info(f"Updated label for path {request.path} to {request.newLabel}")
        return {"message": "History label updated successfully"}
    except Exception as e: logger.error(f"DATABASE ERROR on update: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

@app.get("/history", response_model=List[Dict])
async def get_history():
    if not os.path.exists(DB_FILE): return []
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, session_id, filename, label, timestamp, details_message, path, sharpness, exposure FROM history ORDER BY timestamp DESC").fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Could not retrieve history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve history from database.")

@app.delete("/history")
async def clear_history():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM history")
            conn.commit()
        logger.info("Cleared all records from the history database.")
        for f in os.listdir(HISTORY_IMAGES_DIR):
            try: os.remove(os.path.join(HISTORY_IMAGES_DIR, f))
            except OSError as e: logger.error(f"Error removing history file {f}: {e}")
        return JSONResponse(status_code=200, content={"message": "History and all permanent images cleared successfully."})
    except Exception as e:
        logger.error(f"Failed to clear history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear history.")