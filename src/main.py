from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import imagehash
from PIL import Image
import io
import logging
import os
from fastapi.staticfiles import StaticFiles
from typing import List, Optional

app = FastAPI(title="IMAHE API", description="AI-powered photo sorting API")

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_DIR = "src/images"
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount(f"/static/images", StaticFiles(directory=IMAGE_DIR), name="images")

MAX_PROCESSING_SIZE = (1280, 720)

# --- STATE MANAGEMENT & SETTINGS ---
user_settings = {
    "min_exposure": 50,
    "max_exposure": 200,
    "min_sharpness": 100
}
image_hashes = {}

# --- AI/ML MODEL LOADING ---
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except Exception as e:
    logger.error(f"FATAL: Could not load Haar cascades. Error: {e}")

# --- PYDANTIC MODELS ---
class Settings(BaseModel):
    min_exposure: float
    max_exposure: float
    min_sharpness: float

class FaceDetail(BaseModel):
    reason: str
    count: int

class ClassificationDetails(BaseModel):
    message: Optional[str] = None
    sharpness: Optional[float] = None
    exposure: Optional[float] = None
    face_details: Optional[List[FaceDetail]] = None

class ClassificationResult(BaseModel):
    status: str
    label: str
    details: ClassificationDetails

# --- IMAGE ANALYSIS FUNCTIONS ---
def calculate_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_exposure(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def get_image_hash(image: Image.Image) -> str:
    return str(imagehash.average_hash(image))

def analyze_faces(image: np.ndarray) -> dict:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    total_faces = len(faces)
    closed_eye_faces = 0
    if total_faces == 0:
        return {"is_flagged": False, "total_faces": 0, "closed_eye_faces": 0}
    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) == 0:
            closed_eye_faces += 1
    return {
        "is_flagged": closed_eye_faces > 0,
        "total_faces": total_faces,
        "closed_eye_faces": closed_eye_faces
    }

# --- API ENDPOINTS ---

@app.post("/clear-state/", status_code=204)
async def clear_state():
    global image_hashes
    image_hashes.clear()
    logger.info("In-memory image hash cache has been cleared.")
    return None

@app.post("/upload-image/", response_model=ClassificationResult)
async def upload_image(file: UploadFile = File(...), detect_closed_eyes: bool = Form(False)):
    try:
        contents = await file.read()

        save_path = os.path.join(IMAGE_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_image_for_processing = pil_image.copy()
        pil_image_for_processing.thumbnail(MAX_PROCESSING_SIZE, Image.Resampling.LANCZOS)
        
        # === START OF THE FIX ===
        # The typo cv2.COLOR_RGB_BGR has been corrected to cv2.COLOR_RGB2BGR
        image_cv2_resized = cv2.cvtColor(np.array(pil_image_for_processing), cv2.COLOR_RGB2BGR)
        # === END OF THE FIX ===

        img_hash = get_image_hash(pil_image_for_processing)
        if img_hash in image_hashes:
            logger.info(f"Duplicate found for {file.filename}")
            return ClassificationResult(
                status="success",
                label="Duplicate",
                details=ClassificationDetails(message=f"Duplicate of {image_hashes.get(img_hash, 'unknown')}")
            )
        image_hashes[img_hash] = file.filename

        sharpness = calculate_sharpness(image_cv2_resized)
        exposure = calculate_exposure(image_cv2_resized)
        face_analysis = analyze_faces(image_cv2_resized)
        
        details = ClassificationDetails(
            sharpness=round(sharpness, 2),
            exposure=round(exposure, 2)
        )

        is_bad_quality = (
            sharpness < user_settings["min_sharpness"] or
            exposure < user_settings["min_exposure"] or
            exposure > user_settings["max_exposure"]
        )

        if face_analysis["is_flagged"]:
            face_detail = FaceDetail(
                reason="closed_eyes",
                count=face_analysis["closed_eye_faces"]
            )
            details.face_details = [face_detail]
            
            if detect_closed_eyes:
                label = "Closed Eye"
            else:
                label = "Flagged"
                
        elif is_bad_quality:
            label = "Bad"
            details.message = "Image is blurry or has poor exposure."
        else:
            label = "Good"
            details.message = "Image quality is good."

        logger.info(f"Classified {file.filename} as {label} (Closed Eye Detection: {detect_closed_eyes})")
        
        return ClassificationResult(
            status="success",
            label=label,
            details=details
        )

    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ClassificationResult(
                status="error",
                label="Error",
                details=ClassificationDetails(message=f"An internal error occurred: {str(e)}")
            ).dict()
        )

# Settings and Health Check endpoints
@app.get("/settings/", response_model=Settings)
async def get_settings():
    return user_settings

@app.post("/settings/", response_model=Settings)
async def update_settings(settings: Settings):
    user_settings.update(settings.dict())
    logger.info(f"Settings updated: {user_settings}")
    return user_settings

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}