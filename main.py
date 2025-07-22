import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import json
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Pre-Filter/Scale API with optional Vision output",
    version="1.2.2"
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
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"  # Optional API

def normalize_vision_coordinates(vector_data: Dict, vision_output: Dict) -> Dict:
    """Normalize vision output pixel coordinates to PDF points using page size and image metadata."""
    if not vision_output or "regions" not in vision_output or "image_metadata" not in vision_output:
        return None

    # Extract page dimensions from the first page of vector data
    if not vector_data.get("pages") or not vector_data["pages"]:
        logger.warning("No pages found in vector data, skipping normalization")
        return None
    page_metadata = vector_data["pages"][0]
    page_width_points = page_metadata.get("page_size", {}).get("width", 3370.0)
    page_height_points = page_metadata.get("page_size", {}).get("height", 2384.0)

    # Extract image metadata from vision output
    image_metadata = vision_output["image_metadata"]
    image_width_pixels = image_metadata.get("image_width_pixels", 9969)
    image_height_pixels = image_metadata.get("image_height_pixels", 7052)
    image_dpi_x = image_metadata.get("image_dpi_x", 213.0044)
    image_dpi_y = image_metadata.get("image_dpi_y", 213.0044)

    # Calculate scale factors based on actual dimensions
    scale_x = page_width_points / image_width_pixels if image_width_pixels > 0 else 1.0
    scale_y = page_height_points / image_height_pixels if image_height_pixels > 0 else 1.0

    # Normalize each region's coordinates to PDF points, adjusting for origin
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

        # Clamp coordinates to page boundaries
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
async def process_pdf(file: UploadFile = File(...), vision_output: str = Form(None)):
    """Process PDF: Extract vectors via Vector Drawing API, optionally normalize vision output, and send to Pre-Filter API or Scale API"""
    try:
        # Log the raw input for debugging
        logger.info(f"Received file: {file.filename}")
        logger.info(f"Received vision_output: {vision_output[:500] if vision_output else 'None'}...")

        # Step 1: Call Vector Drawing API with the binary file
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'true',
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
        
        # Parse Vector Drawing API response
        raw_response = vector_response.text
        logger.info(f"Vector Drawing API response length: {len(raw_response)} bytes")
        
        try:
            vector_data = json.loads(raw_response)
            if isinstance(vector_data, str):
                logger.warning("Response is double-encoded, parsing again")
                vector_data = json.loads(vector_data)
            logger.info("Parsed Vector Drawing API response successfully")
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector Drawing API response: {str(e)}")
        
        # Validate structure
        if not isinstance(vector_data, dict) or 'pages' not in vector_data or not vector_data["pages"]:
            raise HTTPException(status_code=400, detail="Vector Drawing API response missing required fields or pages")

        # Determine processing path based on vision_output
        if vision_output:
            # New path: Parse vision_output and normalize coordinates
            try:
                vision_output_data = json.loads(vision_output)
                if isinstance(vision_output_data, list) and vision_output_data:
                    vision_output_dict = vision_output_data[0].get("vision_output", {})
                else:
                    vision_output_dict = vision_output_data
                if not vision_output_dict or not all(k in vision_output_dict for k in ["drawing_type", "scale_api_version", "regions", "image_metadata"]):
                    logger.warning("Invalid vision_output format, falling back to basic processing")
                    vision_output_dict = None
                
                normalized_vision_output = normalize_vision_coordinates(vector_data, vision_output_dict) if vision_output_dict else None
                
                # Send to Pre-Filter API if vision_output is valid
                if normalized_vision_output:
                    combined_data = {
                        "pages": vector_data["pages"],
                        "config": {
                            "min_line_length": 45.0,
                            "keep_all_text": True,
                            "include_diagonal_lines": True,
                            "diagonal_tolerance": 0.1
                        },
                        "vision_output": normalized_vision_output
                    }
                    
                    logger.info("Calling Pre-Filter API with combined data")
                    headers = {'Content-Type': 'application/json'}
                    filter_response = requests.post(
                        PRE_FILTER_API_URL,
                        json=combined_data,
                        headers=headers,
                        timeout=300
                    )
                    
                    if filter_response.status_code != 200:
                        logger.error(f"Pre-Filter API error: {filter_response.status_code} - {filter_response.text}")
                        return {
                            "status": "partial_success",
                            "message": "Vector extraction successful, but pre-filtering failed",
                            "data": vector_data,
                            "filter_error": filter_response.text
                        }
                    
                    filtered_data = filter_response.json()
                    return {
                        "status": "success",
                        "message": "PDF processed with Pre-Filter API",
                        "data": filtered_data
                    }
            except Exception as e:
                logger.error(f"Error processing vision_output: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing vision_output: {str(e)}")

        # Old path: Fall back to Scale API processing if no vision_output or invalid
        logger.info("Falling back to Scale API processing")
        drawings = vector_data["pages"][0].get("drawings", {})
        texts = vector_data["pages"][0].get("texts", [])

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

        temp_file_path = f"/tmp/scale_input_{uuid.uuid4()}.json"
        with open(temp_file_path, 'w') as f:
            json.dump(vector_data_for_scale, f)

        logger.info(f"Saved input for Scale API to {temp_file_path}")
        
        try:
            with open(temp_file_path, 'rb') as f:
                scale_files = {'file': (os.path.basename(temp_file_path), f, 'application/json')}
                logger.info("Calling Scale API")
                scale_response = requests.post("https://scale-api-69gl.onrender.com/extract-scale/", files=scale_files, timeout=300)
            
            os.unlink(temp_file_path)
            
            if scale_response.status_code != 200:
                logger.error(f"Scale API error: {scale_response.status_code} - {scale_response.text}")
                return {
                    "vector_data": vector_data,
                    "scale_data": None,
                    "error": scale_response.text
                }
            
            scale_data = scale_response.json()
            return {
                "vector_data": vector_data,
                "scale_data": scale_data,
                "timestamp": "2025-07-18"
            }
        except Exception as e:
            logger.error(f"Error calling Scale API: {e}")
            return {
                "vector_data": vector_data,
                "scale_data": None,
                "error": f"Error calling Scale API: {str(e)}"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.2.2"}

@app.get("/")
async def root():
    return {
        "title": "Master API",
        "description": "Processes PDFs through Vector Drawing API and Pre-Filter/Scale API",
        "endpoints": {
            "/": "This page",
            "/process/": "POST - Process PDF file",
            "/health/": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
