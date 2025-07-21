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

app = FastAPI(title="IMAHE API", description="AI-powered photo sorting API")

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the folder for saved images if it doesn't exist
IMAGE_DIR = "src/images"
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount(f"/static/images", StaticFiles(directory=IMAGE_DIR), name="images")

# --- STATE MANAGEMENT & SETTINGS ---

# In-memory storage for settings and image hashes.
# The 'image_hashes' dictionary is the primary cause of the "all duplicates" issue.
# We will add an endpoint to clear it before each batch.
user_settings = {
    "min_exposure": 50,
    "max_exposure": 200,
    "min_sharpness": 100
}
image_hashes = {}

# --- AI/ML MODEL LOADING ---

# ✅ FIX: Load both face and eye detectors. Detecting a face first makes eye detection more reliable.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except Exception as e:
    logger.error(f"FATAL: Could not load Haar cascades. Ensure OpenCV is correctly installed. Error: {e}")
    # In a real app, you might want to exit if essential models don't load.

# --- PYDANTIC MODELS ---

class Settings(BaseModel):
    min_exposure: float
    max_exposure: float
    min_sharpness: float

class ClassificationResult(BaseModel):
    status: str
    label: str
    details: dict

# --- IMAGE ANALYSIS FUNCTIONS ---

def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_exposure(image: np.ndarray) -> float:
    """Calculate image exposure based on average brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def get_image_hash(image: Image.Image) -> str:
    """Generate perceptual hash for an image."""
    return str(imagehash.average_hash(image))

def detect_closed_eyes(image: np.ndarray) -> bool:
    """
    ✅ FIX: Improved closed-eye detection.
    This new logic first finds faces, then looks for eyes within each face.
    If a face is found but no eyes are, it's a strong indicator of closed eyes.
    This prevents non-portrait photos (landscapes, etc.) from being misclassified.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return False  # No faces found, so it can't be a "closed eye" portrait.

    for (x, y, w, h) in faces:
        # Create a Region of Interest (ROI) for the face
        roi_gray = gray_image[y:y+h, x:x+w]
        # Detect eyes within the face's ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) == 0:
            # If we found a face but no eyes inside it, it's highly likely they are closed.
            return True

    return False # Faces were found, and eyes were found within them.

# --- API ENDPOINTS ---

@app.post("/clear-state/", status_code=204)
async def clear_state():
    """
    ✅ NEW ENDPOINT: Clears the in-memory image hash set.
    The Blazor client should call this endpoint BEFORE starting a new processing batch
    to prevent images from a previous run from being marked as duplicates.
    """
    global image_hashes
    image_hashes.clear()
    logger.info("In-memory image hash cache has been cleared.")
    return None # Return a 204 No Content response

@app.post("/upload-image/", response_model=ClassificationResult)
async def upload_image(file: UploadFile = File(...)):
    """Upload and classify an image as Good, Bad, Duplicate, or Closed Eye."""
    try:
        # 1. Read and prepare the image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB") # Convert to RGB to handle PNGs with alpha, etc.
        image_np = np.array(pil_image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 2. Save the uploaded image so the Blazor client can display it
        save_path = os.path.join(IMAGE_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        # 3. Check for duplicates
        img_hash = get_image_hash(pil_image)
        if img_hash in image_hashes:
            logger.info(f"Duplicate found for {file.filename} with hash {img_hash}")
            return ClassificationResult(
                status="success",
                label="Duplicate",
                details={"message": f"Duplicate of {image_hashes[img_hash]}"}
            )
        # Add the hash to our cache *after* the check
        image_hashes[img_hash] = file.filename

        # 4. Perform classifications in a logical order
        sharpness = calculate_sharpness(image_cv2)
        exposure = calculate_exposure(image_cv2)
        
        details = {
            "sharpness": round(sharpness, 2),
            "exposure": round(exposure, 2)
        }

        # Check for technical flaws first
        is_bad_quality = (
            sharpness < user_settings["min_sharpness"] or
            exposure < user_settings["min_exposure"] or
            exposure > user_settings["max_exposure"]
        )

        # Check for content flaws (closed eyes)
        has_closed_eyes = detect_closed_eyes(image_cv2)

        # 5. Determine the final label based on the checks
        if has_closed_eyes:
            label = "Closed Eye"
            details["reason"] = "A face was detected with no visible eyes."
        elif is_bad_quality:
            label = "Bad"
            details["reason"] = "Image is blurry or has poor exposure."
        else:
            label = "Good"

        logger.info(f"Classified {file.filename} as {label} with details: {details}")
        
        return ClassificationResult(
            status="success",
            label=label,
            details=details
        )

    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}", exc_info=True)
        # Return a specific error structure that the frontend can handle
        return JSONResponse(
            status_code=500,
            content=ClassificationResult(
                status="error",
                label="Error",
                details={"message": f"An internal error occurred: {str(e)}"}
            ).dict()
        )

# Settings and Health Check endpoints (no changes needed to these)
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