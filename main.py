import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Scale API, with optional vision output",
    version="1.0.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_API_URL = "https://vector-drawing.onrender.com/extract/"  # Corrected URL
SCALE_API_URL = "https://scale-api-69gl.onrender.com/extract-scale/"

def normalize_vision_coordinates(vector_data: Dict, vision_output: Dict) -> Dict:
    """Normalize vision output pixel coordinates to PDF points using page size and image metadata."""
    if not vision_output or "regions" not in vision_output or "image_metadata" not in vision_output:
        return None

    if not vector_data.get("pages") or not vector_data["pages"]:
        logger.warning("No pages found in vector data, skipping normalization")
        return None
    page_metadata = vector_data["pages"][0]
    page_width_points = page_metadata.get("page_size", {}).get("width", 3370.0)
    page_height_points = page_metadata.get("page_size", {}).get("height", 2384.0)

    image_metadata = vision_output["image_metadata"]
    image_width_pixels = image_metadata.get("image_width_pixels", 9969)
    image_height_pixels = image_metadata.get("image_height_pixels", 7052)

    scale_x = page_width_points / image_width_pixels if image_width_pixels > 0 else 1.0
    scale_y = page_height_points / image_height_pixels if image_height_pixels > 0 else 1.0

    normalized_regions = []
    for region in vision_output["regions"]:
        coords = region["coordinate_block"]
        if len(coords) != 4 or any(c < 0 for c in coords):
            logger.warning(f"Invalid coordinates for region {region['label']}, skipping")
            continue

        x0_points = coords[0] * scale_x
        y0_points = page_height_points - (coords[1] * scale_y)
        x1_points = coords[2] * scale_x
        y1_points = page_height_points - (coords[3] * scale_y)

        x0_points = max(0, min(x0_points, page_width_points))
        x1_points = max(0, min(x1_points, page_width_points))
        y0_points = max(0, min(y0_points, page_height_points))
        y1_points = max(0, min(y1_points, page_height_points))

        normalized_regions.append({
            "coordinate_block": [x0_points, y0_points, x1_points, y1_points],
            "label": region["label"]
        })

    return {
        "drawing_type": vision_output["drawing_type"],
        "scale_api_version": vision_output["scale_api_version"],
        "regions": normalized_regions,
        "image_metadata": image_metadata
    }

@app.post("/process/")
async def process_pdf(file: UploadFile, vision_output: str = Form(None)):
    """Process PDF: Extract vectors via Vector Drawing API, optionally normalize vision output, and send to Scale API"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Optional vision_output handling
        vision_data = None
        normalized_vision = None
        if vision_output:
            try:
                vision_data = json.loads(vision_output)
                logger.info("Vision output parsed successfully")
                normalized_vision = normalize_vision_coordinates({"pages": [{"page_size": {"width": 3370.0, "height": 2384.0}}]}, vision_data)
                if normalized_vision:
                    logger.info("Vision coordinates normalized successfully")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for vision_output: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")

        # Step 1: Call Vector Drawing API with specified parameters
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'false',
            'remove_non_essential': 'false',
            'precision': '2'
        }
        
        logger.info("Calling Vector Drawing API")
        vector_response = requests.post(VECTOR_API_URL, files=files, params=params, timeout=300)
        
        if vector_response.status_code != 200:
            raise HTTPException(
                status_code=vector_response.status_code,
                detail=f"Vector Drawing API error: {vector_response.text}"
            )
        
        # Get the raw response text
        raw_response = vector_response.text
        logger.info(f"Raw Vector Drawing API response (first 100 chars): {raw_response[:100]}")
        logger.info(f"Response length: {len(raw_response)} bytes")
        
        # Don't use requests.json() - explicitly parse the JSON string
        try:
            vector_data = json.loads(raw_response)
            logger.info(f"JSON parsed successfully, type={type(vector_data)}")
            logger.info(f"Top-level keys: {list(vector_data.keys()) if isinstance(vector_data, dict) else 'Not a dict'}")
            
            if isinstance(vector_data, str):
                logger.warning("Parsed result is still a string, attempting to parse again")
                vector_data = json.loads(vector_data)
                logger.info(f"Second parse: type={type(vector_data)}")
                logger.info(f"Second parse keys: {list(vector_data.keys()) if isinstance(vector_data, dict) else 'Not a dict'}")
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            debug_path = f"/tmp/vector_response_{uuid.uuid4()}.json"
            with open(debug_path, 'w') as f:
                f.write(raw_response)
            logger.info(f"Saved problematic response to {debug_path}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector Drawing API response: {str(e)}")
        
        if not isinstance(vector_data, dict):
            raise HTTPException(status_code=400, detail="Parsed result is not a dictionary")
            
        if 'pages' not in vector_data or not vector_data['pages']:
            raise HTTPException(status_code=400, detail="No pages found in vector data")
            
        # Get the first page
        page = vector_data['pages'][0]
        
        # Extract drawings and texts
        drawings = page.get('drawings', {})
        texts = page.get('texts', [])
        
        logger.info(f"Found {len(texts)} texts and drawing types: {list(drawings.keys())}")
        
        # Prepare data for Scale API
        vector_data_for_scale = {
            "vector_data": [],
            "texts": []
        }
        
        if 'lines' in drawings and isinstance(drawings['lines'], list):
            for line in drawings['lines']:
                if 'p1' in line and 'p2' in line:
                    vector_data_for_scale["vector_data"].append({
                        "type": line.get("type", "line"),
                        "p1": line["p1"],
                        "p2": line["p2"],
                        "length": line.get("length")
                    })
        
        for text in texts:
            if 'text' in text and 'position' in text:
                vector_data_for_scale["texts"].append({
                    "text": text["text"],
                    "position": text["position"]
                })
        
        logger.info(f"Prepared {len(vector_data_for_scale['vector_data'])} vectors and {len(vector_data_for_scale['texts'])} texts for Scale API")
        
        # Save to temporary file
        temp_file_path = f"/tmp/scale_input_{uuid.uuid4()}.json"
        with open(temp_file_path, 'w') as f:
            json.dump(vector_data_for_scale, f)
        
        logger.info(f"Saved input for Scale API to {temp_file_path}")
        
        # Call Scale API
        try:
            with open(temp_file_path, 'rb') as f:
                scale_files = {'file': (os.path.basename(temp_file_path), f, 'application/json')}
                logger.info("Calling Scale API")
                scale_response = requests.post(SCALE_API_URL, files=scale_files, timeout=300)
            
            os.unlink(temp_file_path)
            
            if scale_response.status_code != 200:
                logger.warning(f"Scale API error: {scale_response.status_code}")
                return {
                    "vector_data": vector_data,
                    "scale_data": None,
                    "error": scale_response.text,
                    "vision_data": vision_data if vision_data else None
                }
            
            scale_data = scale_response.json()
            logger.info("Scale API response parsed successfully")
            
            return {
                "vector_data": vector_data,
                "scale_data": scale_data,
                "timestamp": "2025-07-18",
                "vision_data": vision_data if vision_data else None,
                "normalized_vision": normalized_vision if normalized_vision else None
            }
            
        except Exception as e:
            logger.error(f"Error calling Scale API: {e}")
            return {
                "vector_data": vector_data,
                "scale_data": None,
                "error": f"Error calling Scale API: {str(e)}",
                "vision_data": vision_data if vision_data else None
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0.1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
