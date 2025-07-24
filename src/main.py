from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import imagehash
from PIL import Image
import io # Required for in-memory byte handling
import logging
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="IMAHE API", description="AI-powered photo sorting API")

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_DIR = "src/images"
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount(f"/static/images", StaticFiles(directory=IMAGE_DIR), name="images")

# --- NEW: Define a max size for faster processing ---
MAX_PROCESSING_SIZE = (1280, 720) # (Width, Height) - Balances speed and accuracy

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

# --- PYDANTIC MODELS (No changes needed) ---
class Settings(BaseModel):
    min_exposure: float
    max_exposure: float
    min_sharpness: float

class ClassificationResult(BaseModel):
    status: str
    label: str
    details: dict

# --- IMAGE ANALYSIS FUNCTIONS (No changes needed) ---
def calculate_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_exposure(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def get_image_hash(image: Image.Image) -> str:
    return str(imagehash.average_hash(image))

def detect_closed_eyes(image: np.ndarray) -> bool:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return False

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) == 0:
            return True

    return False

# --- API ENDPOINTS ---

@app.post("/clear-state/", status_code=204)
async def clear_state():
    """
    Clears the in-memory image hash set. Call before starting a new batch.
    NOTE: When using multiple workers (Gunicorn), this clears the cache for whichever
    worker receives the request. This is an acceptable trade-off for simplicity.
    """
    global image_hashes
    image_hashes.clear()
    logger.info("In-memory image hash cache has been cleared.")
    return None

@app.post("/upload-image/", response_model=ClassificationResult)
async def upload_image(file: UploadFile = File(...)):
    """Upload, resize for speed, and classify an image."""
    try:
        # 1. Read image bytes
        contents = await file.read()

        # 2. Save the ORIGINAL full-quality image for the UI to display
        save_path = os.path.join(IMAGE_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        # === START: NEW HIGH-SPEED RESIZING LOGIC ===
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Create a copy for resizing, leaving the original pil_image if needed
        pil_image_for_processing = pil_image.copy()
        
        # Resize the image if it's larger than our max size
        pil_image_for_processing.thumbnail(MAX_PROCESSING_SIZE, Image.Resampling.LANCZOS)
        
        # Convert the resized PIL image to the OpenCV format needed for analysis
        image_cv2_resized = cv2.cvtColor(np.array(pil_image_for_processing), cv2.COLOR_RGB2BGR)
        # === END: NEW HIGH-SPEED RESIZING LOGIC ===


        # 3. Check for duplicates using the resized image (it's faster)
        img_hash = get_image_hash(pil_image_for_processing)
        if img_hash in image_hashes:
            logger.info(f"Duplicate found for {file.filename}")
            return ClassificationResult(
                status="success",
                label="Duplicate",
                details={"message": f"Duplicate of {image_hashes.get(img_hash, 'unknown')}"}
            )
        image_hashes[img_hash] = file.filename

        # 4. Perform all classifications on the SMALLER, RESIZED image
        sharpness = calculate_sharpness(image_cv2_resized)
        exposure = calculate_exposure(image_cv2_resized)
        has_closed_eyes = detect_closed_eyes(image_cv2_resized)
        
        details = {
            "sharpness": round(sharpness, 2),
            "exposure": round(exposure, 2)
        }

        is_bad_quality = (
            sharpness < user_settings["min_sharpness"] or
            exposure < user_settings["min_exposure"] or
            exposure > user_settings["max_exposure"]
        )

        # 5. Determine the final label
        if has_closed_eyes:
            label = "Closed Eye"
            details["reason"] = "A face was detected with no visible eyes."
        elif is_bad_quality:
            label = "Bad"
            details["reason"] = "Image is blurry or has poor exposure."
        else:
            label = "Good"

        logger.info(f"Classified {file.filename} as {label}")
        
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
                details={"message": f"An internal error occurred: {str(e)}"}
            ).dict()
        )

# Settings and Health Check endpoints (no changes needed)
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