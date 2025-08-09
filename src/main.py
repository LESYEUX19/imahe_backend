from fastapi import FastAPI, UploadFile, File, HTTPException
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

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IMAHE API", description="AI-powered photo sorting API")

IMAGE_DIR = "src/images"
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount(f"/static/images", StaticFiles(directory=IMAGE_DIR), name="images")

MAX_PROCESSING_SIZE = (1280, 720)

# --- STATE MANAGEMENT & SETTINGS ---
user_settings = {
    "min_exposure": 50.0,
    "max_exposure": 200.0,
    "min_sharpness": 100.0
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

class ClassificationDetails(BaseModel):
    message: Optional[str] = None
    sharpness: Optional[float] = None
    exposure: Optional[float] = None

class ClassificationResult(BaseModel):
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
    if len(faces) == 0:
        return {"has_closed_eyes": False, "closed_eye_count": 0}
    closed_eye_count = 0
    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) == 0:
            closed_eye_count += 1
    return {
        "has_closed_eyes": closed_eye_count > 0,
        "closed_eye_count": closed_eye_count
    }

# --- API ENDPOINTS ---

@app.get("/settings/", response_model=Settings)
async def get_settings():
    """Retrieves the current classification settings."""
    logger.info("GET /settings/ - Retrieving current settings.")
    return user_settings

@app.post("/settings/", response_model=Settings)
async def update_settings(new_settings: Settings):
    """Updates the classification settings."""
    global user_settings
    user_settings = new_settings.dict()
    logger.info(f"POST /settings/ - Settings updated to: {user_settings}")
    return user_settings

@app.post("/clear-state/", status_code=204)
async def clear_state():
    """Clears the in-memory cache of image hashes."""
    global image_hashes
    image_hashes.clear()
    logger.info("In-memory image hash cache has been cleared.")
    return None

@app.post("/upload-image/", response_model=ClassificationResult)
async def upload_image(file: UploadFile = File(...)):
    """Processes a single image and returns its classification."""
    try:
        contents = await file.read()
        save_path = os.path.join(IMAGE_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_image_for_processing = pil_image.copy()
        pil_image_for_processing.thumbnail(MAX_PROCESSING_SIZE, Image.Resampling.LANCZOS)
        
        image_cv2_resized = cv2.cvtColor(np.array(pil_image_for_processing), cv2.COLOR_RGB2BGR)

        # --- Classification Logic with Fixed Priority ---

        # 1. Check for Duplicate (Highest Priority)
        img_hash = get_image_hash(pil_image_for_processing)
        if img_hash in image_hashes:
            return ClassificationResult(
                label="Duplicate",
                details=ClassificationDetails(message=f"Duplicate of {image_hashes[img_hash]}")
            )
        image_hashes[img_hash] = file.filename

        # If not a duplicate, perform all other analyses
        sharpness = calculate_sharpness(image_cv2_resized)
        exposure = calculate_exposure(image_cv2_resized)
        face_analysis = analyze_faces(image_cv2_resized)
        details = ClassificationDetails() # Start with an empty details object

        # 2. Check for Closed Eye
        if face_analysis["has_closed_eyes"]:
            label = "Closed Eye"
            details.message = f"Detected {face_analysis['closed_eye_count']} face(s) with closed eyes."
        
        # 3. Check for Blurred
        elif sharpness < user_settings["min_sharpness"]:
            label = "Blurred"
            details.sharpness = round(sharpness, 2) # Add sharpness value
            details.message = f"Image is blurry (Sharpness: {details.sharpness})."
            
        # 4. If all checks pass, it's Good
        else:
            label = "Good"
            details.sharpness = round(sharpness, 2) # Add sharpness value
            details.exposure = round(exposure, 2)  # Add exposure value
            details.message = "Image quality meets thresholds."

        logger.info(f"Classified {file.filename} as {label}")
        
        return ClassificationResult(
            label=label,
            details=details
        )

    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}", exc_info=True)
        # Return a JSONResponse for errors to match the expected schema
        return JSONResponse(
            status_code=500,
            content={
                "label": "Error",
                "details": {"message": str(e)}
            }
        )