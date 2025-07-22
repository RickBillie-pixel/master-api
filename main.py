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
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API with Vision to PDF coordinate mapping",
    version="1.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_API_URL = "https://vector-drawing.onrender.com/extract/"
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

def normalize_vision_coordinates(vector_data: Dict, vision_output: Dict) -> Dict:
    """Normalize vision output pixel coordinates to PDF points using page size and image metadata."""
    if not vision_output or "regions" not in vision_output or "image_metadata" not in vision_output:
        return vision_output

    # Extract page dimensions from the first page of vector data
    if not vector_data.get("pages") or not vector_data["pages"]:
        raise ValueError("No pages found in vector data")
    page_metadata = vector_data["pages"][0]
    page_width_points = page_metadata.get("page_size", {}).get("width", 3370.0)  # Default based on your example
    page_height_points = page_metadata.get("page_size", {}).get("height", 2384.0)  # Default based on your example

    # Extract image metadata from vision output
    image_metadata = vision_output["image_metadata"]
    image_width_pixels = image_metadata.get("image_width_pixels", 9969)  # Default from your example
    image_height_pixels = image_metadata.get("image_height_pixels", 7052)  # Default from your example
    image_dpi_x = image_metadata.get("image_dpi_x", 213.0044)  # Default from your example
    image_dpi_y = image_metadata.get("image_dpi_y", 213.0044)  # Default from your example

    # Calculate scale factors based on actual dimensions (pixels to points)
    scale_x = page_width_points / image_width_pixels if image_width_pixels > 0 else 1.0
    scale_y = page_height_points / image_height_pixels if image_height_pixels > 0 else 1.0

    # Normalize each region's coordinates to PDF points, adjusting for origin (top-left to bottom-left)
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

        normalized_regions.append({
            "coordinate_block": [x0_points, y0_points, x1_points, y1_points],
            "label": region["label"]
        })

    return {
        "drawing_type": vision_output["drawing_type"],
        "scale_api_version": vision_output["scale_api_version"],
        "regions": normalized_regions,
        "image_metadata": image_metadata  # Preserve metadata
    }

@app.post("/process/")
async def process_pdf(file: UploadFile, vision_output: str = Form(...)):
    """Process PDF: Extract vectors via Vector Drawing API, normalize vision output, and send to Pre-Filter API"""
    filtered_data = None
    vector_data = None
    
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Parse vision_output (handle the array structure from n8n)
        try:
            vision_output_list = json.loads(vision_output)
            if not isinstance(vision_output_list, list) or not vision_output_list:
                raise ValueError("Invalid vision output: expected a non-empty array")
            # Extract the first content item (assuming single JSON response)
            content = vision_output_list[0].get("content", "")
            if not content or not content.startswith("```json\n") or not content.endswith("\n```"):
                raise ValueError("Invalid vision output format: expected JSON code block")
            vision_output_dict = json.loads(content[7:-4])  # Remove ```json\n and \n```
            logger.info("Vision output parsed successfully")
            if not all(k in vision_output_dict for k in ["drawing_type", "scale_api_version", "regions"]):
                raise ValueError("Invalid vision output format: missing required fields")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing or validation error for vision_output: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")
        
        # Step 2: Call Vector Drawing API
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
        
        # Step 3: Normalize vision coordinates to match vector data's PDF coordinate space
        normalized_vision_output = normalize_vision_coordinates(vector_data, vision_output_dict)
        
        # Step 4: Combine vector drawing output and normalized vision output for Pre-Filter API
        combined_data = {
            "pages": vector_data["pages"],  # Raw vector data in pdf_coordinate_space (unchanged)
            "config": {
                "min_line_length": 45.0,
                "keep_all_text": True,
                "include_diagonal_lines": True,
                "diagonal_tolerance": 0.1
            },
            "vision_output": normalized_vision_output  # Normalized to pdf_coordinate_space
        }
        
        logger.info("Data combined successfully: vector output and normalized vision output")
        
        # Step 5: Send to Pre-Filter API
        logger.info("Calling Pre-Filter API with combined data")
        headers = {'Content-Type': 'application/json'}
        filter_response = requests.post(
            PRE_FILTER_API_URL,
            json=combined_data,
            headers=headers,
            timeout=300
        )
        
        logger.info(f"Pre-Filter API response status: {filter_response.status_code}")
        
        if filter_response.status_code != 200:
            logger.error(f"Pre-Filter API error: {filter_response.status_code} - {filter_response.text}")
            return {
                "status": "partial_success",
                "message": "Vector extraction successful, but pre-filtering failed",
                "data": combined_data,
                "filter_error": filter_response.text
            }
        
        # Parse Pre-Filter response
        try:
            filtered_data = filter_response.json()
            logger.info("Pre-Filter API response parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing Pre-Filter API response: {e}")
            return {
                "status": "partial_success",
                "message": "Vector extraction successful, but pre-filter response parsing failed",
                "data": combined_data,
                "filter_error": str(e)
            }
        
        # Return success
        return {
            "status": "success",
            "message": "PDF processed successfully through both APIs",
            "data": filtered_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.1.0"}

@app.get("/")
async def root():
    return {
        "title": "Master API",
        "description": "Processes PDFs through Vector Drawing API and Pre-Filter API",
        "endpoints": {
            "/": "This page",
            "/process/": "POST - Process PDF file",
            "/health/": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
