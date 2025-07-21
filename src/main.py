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
import glob
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="IMAHE API", description="AI-powered photo sorting API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for settings and image hashes (replace with DB in production)
user_settings = {
    "min_exposure": 50,
    "max_exposure": 200,
    "min_sharpness": 100
}
image_hashes = {}

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Mount static files for images
app.mount("/static/images", StaticFiles(directory="src/images"), name="images")

# Pydantic model for settings
class Settings(BaseModel):
    min_exposure: float
    max_exposure: float
    min_sharpness: float
    
    class Config:
        schema_extra = {
            "example": {
                "min_exposure": 50.0,
                "max_exposure": 200.0,
                "min_sharpness": 100.0
            }
        }

# Pydantic model for partial settings updates
class PartialSettings(BaseModel):
    min_exposure: float = None
    max_exposure: float = None
    min_sharpness: float = None
    
    class Config:
        schema_extra = {
            "example": {
                "min_exposure": 60.0,
                "max_exposure": 180.0
            }
        }

# Pydantic model for image classification response
class ClassificationResult(BaseModel):
    status: str
    label: str
    details: dict

def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_exposure(image: np.ndarray) -> float:
    """Calculate image exposure based on brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def get_image_hash(image: Image.Image) -> str:
    """Generate perceptual hash for an image."""
    return str(imagehash.average_hash(image))

def detect_closed_eyes(image: np.ndarray) -> bool:
    """Detect if there are closed eyes in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) == 0  # If no eyes are detected, we assume closed eyes

@app.post("/upload-image/", response_model=ClassificationResult)
async def upload_image(file: UploadFile = File(...)):
    """Upload and classify an image as Good, Bad, Duplicate, or Closed Eye."""
    try:
        # Read image
        contents = await file.read()
        image = np.array(Image.open(io.BytesIO(contents)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Save image to images/ folder
        save_path = os.path.join("src/images", file.filename)
        os.makedirs("src/images", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(contents)

        # Check for duplicates
        pil_image = Image.open(io.BytesIO(contents))
        img_hash = get_image_hash(pil_image)
        if img_hash in image_hashes:
            return ClassificationResult(
                status="success",
                label="Duplicate",
                details={"message": "Image is a duplicate"}
            )
        image_hashes[img_hash] = file.filename

        # Calculate image metrics
        sharpness = calculate_sharpness(image)
        exposure = calculate_exposure(image)

        # Log sharpness and exposure for debugging
        logger.info(f"Sharpness: {sharpness}, Exposure: {exposure}")

        # Check for closed eyes
        if detect_closed_eyes(image):
            label = "Closed Eye"
            details = {
                "sharpness": sharpness,
                "exposure": exposure
            }
        else:
            # Classify image as Good or Bad
            details = {
                "sharpness": sharpness,
                "exposure": exposure
            }

            if (sharpness < user_settings["min_sharpness"] or
                  exposure < user_settings["min_exposure"] or
                  exposure > user_settings["max_exposure"]):
                label = "Bad"
                details["reason"] = "Low quality (sharpness or exposure out of range)"
            else:
                label = "Good"

        return ClassificationResult(
            status="success",
            label=label,
            details=details
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return ClassificationResult(
            status="error",
            label="Error",
            details={"message": str(e)}
        )

@app.get("/settings/", response_model=Settings)
async def get_settings():
    """Retrieve current user settings."""
    return user_settings

@app.post("/settings/", response_model=Settings)
async def update_settings(settings: Settings):
    """Update user settings for image classification."""
    try:
        # Validate that max_exposure is greater than min_exposure
        if settings.max_exposure <= settings.min_exposure:
            raise HTTPException(
                status_code=400, 
                detail="max_exposure must be greater than min_exposure"
            )
        
        # Validate that values are positive
        if settings.min_exposure < 0 or settings.max_exposure < 0 or settings.min_sharpness < 0:
            raise HTTPException(
                status_code=400,
                detail="All values must be positive"
            )
        
        user_settings.update(settings.dict())
        logger.info(f"Settings updated: {user_settings}")
        return user_settings
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/settings/", response_model=Settings)
async def update_settings_partial(partial_settings: PartialSettings):
    """Update user settings partially (only provided fields)."""
    try:
        # Get current settings
        current_settings = user_settings.copy()
        
        # Update only provided fields
        update_data = {k: v for k, v in partial_settings.dict().items() if v is not None}
        current_settings.update(update_data)
        
        # Validate the updated settings
        if current_settings["max_exposure"] <= current_settings["min_exposure"]:
            raise HTTPException(
                status_code=400, 
                detail="max_exposure must be greater than min_exposure"
            )
        
        # Validate that values are positive
        if (current_settings["min_exposure"] < 0 or 
            current_settings["max_exposure"] < 0 or 
            current_settings["min_sharpness"] < 0):
            raise HTTPException(
                status_code=400,
                detail="All values must be positive"
            )
        
        user_settings.update(current_settings)
        logger.info(f"Settings partially updated: {user_settings}")
        return user_settings
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Check API health."""
    return {"status": "healthy"}

@app.get("/images/organized/")
async def get_organized_images():
    """List all images in src/images/ organized as good, bad, duplicate, or closed eye, with URLs."""
    images_dir = os.path.join("src", "images")
    if not os.path.exists(images_dir):
        return {"good": [], "bad": [], "duplicate": [], "closed_eye": []}

    files = glob.glob(os.path.join(images_dir, "*"))
    seen_hashes = set()
    good, bad, duplicate, closed_eye = [], [], [], []

    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                contents = f.read()
            pil_image = Image.open(io.BytesIO(contents))
            image = np.array(pil_image)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_hash = get_image_hash(pil_image)
            filename = os.path.basename(file_path)
            url = f"/static/images/{filename}"

            if img_hash in seen_hashes:
                duplicate.append({
                    "filename": filename,
                    "url": url,
                    "message": "Image is a duplicate"
                })
                continue
            seen_hashes.add(img_hash)

            sharpness = calculate_sharpness(image)
            exposure = calculate_exposure(image)
            details = {
                "filename": filename,
                "url": url,
                "sharpness": sharpness,
                "exposure": exposure
            }

            # Check for closed eyes
            if detect_closed_eyes(image):
                closed_eye.append(details)
            elif (sharpness < user_settings["min_sharpness"] or
                  exposure < user_settings["min_exposure"] or
                  exposure > user_settings["max_exposure"]):
                details["reason"] = "Low quality (sharpness or exposure out of range)"
                bad.append(details)
            else:
                good.append(details)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    return {"good": good, "bad": bad, "duplicate": duplicate, "closed_eye": closed_eye}
