# IMAHE API

AI-powered photo sorting backend using FastAPI. This API lets you upload images, automatically classifies them as Good, Bad, or Duplicate based on sharpness and exposure, and provides endpoints to manage settings and organize your photo collection.

## Features
- **Upload images** and get instant quality classification
- **Detect duplicates** using perceptual hashing
- **Customizable thresholds** for sharpness and exposure
- **Organize images** by quality (Good, Bad, Duplicate)
- **Get direct URLs** to your images

## Requirements
- Python 3.8, 3.9, 3.10, or 3.11 (not 3.13!)
- pip
- Windows, Linux, or MacOS

## Installation & Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/imahe_api.git
   cd imahe_api
   ```

2. **Create a virtual environment (recommended)**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**
   ```sh
   pip install fastapi uvicorn opencv-python numpy imagehash pillow
   ```
   > **Note:** If you have trouble installing `dlib` or other packages, see Troubleshooting below.

4. **Run the server**
   ```sh
   uvicorn src.main:app --reload
   ```
   The API will be available at [http://localhost:8000](http://localhost:8000)

## API Endpoints

### 1. **Upload and Classify Image**
- **POST** `/upload-image/`
- **Body:** `multipart/form-data` with an image file
- **Response:**
  ```json
  {
    "status": "success",
    "label": "Good",
    "details": {
      "sharpness": 120.5,
      "exposure": 110.2
    }
  }
  ```

### 2. **Get Current Settings**
- **GET** `/settings/`
- **Response:**
  ```json
  {
    "min_exposure": 50.0,
    "max_exposure": 200.0,
    "min_sharpness": 100.0
  }
  ```

### 3. **Update All Settings**
- **POST** `/settings/`
- **Body:**
  ```json
  {
    "min_exposure": 60.0,
    "max_exposure": 180.0,
    "min_sharpness": 120.0
  }
  ```

### 4. **Update Some Settings**
- **PATCH** `/settings/`
- **Body:**
  ```json
  {
    "min_sharpness": 150.0
  }
  ```

### 5. **Health Check**
- **GET** `/health/`
- **Response:** `{ "status": "healthy" }`

### 6. **Organize All Images**
- **GET** `/images/organized/`
- **Response:**
  ```json
  {
    "good": [
      {
        "filename": "img1.jpg",
        "url": "/static/images/img1.jpg",
        "sharpness": 135.2,
        "exposure": 120.5
      }
    ],
    "bad": [
      {
        "filename": "img2.jpg",
        "url": "/static/images/img2.jpg",
        "sharpness": 45.8,
        "exposure": 30.2,
        "reason": "Low quality (sharpness or exposure out of range)"
      }
    ],
    "duplicate": [
      {
        "filename": "img3.jpg",
        "url": "/static/images/img3.jpg",
        "message": "Image is a duplicate"
      }
    ]
  }
  ```

### 7. **Access Image Files**
- **GET** `/static/images/{filename}`
- Direct URL to each image file.

## Usage Tips
- Use the built-in Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs) to test all endpoints interactively.
- Images are stored in `src/images/`.
- Settings are in-memory (reset on server restart).

## Troubleshooting
- **TensorFlow or dlib install errors:**
  - Use Python 3.10 or 3.11 (not 3.13)
  - For `dlib`, download a pre-built wheel from [Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib) and install with `pip install path/to/whl`.
- **Port already in use:** Change the port with `uvicorn src.main:app --reload --port 8080`
- **Images not showing:** Make sure you use the `/static/images/filename` URL and the file exists in `src/images/`.

## License
MIT 