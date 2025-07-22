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
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API with vision output integration",
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

# UPDATED: Pre-Filter API instead of Scale API
VECTOR_API_URL = "https://vector-drawing.onrender.com/extract/"
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

def convert_image_coords_to_pdf(vision_data: Dict, pdf_dimensions: Dict) -> Dict:
    """Convert image pixel coordinates to PDF point coordinates"""
    
    if not vision_data or not pdf_dimensions:
        logger.warning("Missing vision data or PDF dimensions for coordinate conversion")
        return vision_data
    
    try:
        # Get image metadata
        image_meta = vision_data.get("image_metadata", {})
        image_width_px = image_meta.get("image_width_pixels", 1)
        image_height_px = image_meta.get("image_height_pixels", 1)
        image_dpi = image_meta.get("image_dpi_x", 213.0)
        
        # Get PDF dimensions from coordinate bounds or fallback
        pdf_bounds = pdf_dimensions.get("coordinate_bounds", {})
        pdf_width = pdf_bounds.get("width", pdf_bounds.get("max_x", 595))
        pdf_height = pdf_bounds.get("height", pdf_bounds.get("max_y", 842))
        
        logger.info(f"Converting coordinates: Image {image_width_px}x{image_height_px}px @ {image_dpi}DPI to PDF {pdf_width}x{pdf_height}pts")
        
        # Calculate scale factors
        scale_x = pdf_width / image_width_px
        scale_y = pdf_height / image_height_px
        
        # Convert regions
        converted_regions = []
        for region in vision_data.get("regions", []):
            coord_block = region.get("coordinate_block", [])
            if len(coord_block) >= 4:
                # Convert [x1, y1, x2, y2] from image pixels to PDF points
                x1_pdf = coord_block[0] * scale_x
                y1_pdf = coord_block[1] * scale_y
                x2_pdf = coord_block[2] * scale_x
                y2_pdf = coord_block[3] * scale_y
                
                converted_region = region.copy()
                converted_region["coordinate_block_pdf"] = [
                    round(x1_pdf, 2), 
                    round(y1_pdf, 2), 
                    round(x2_pdf, 2), 
                    round(y2_pdf, 2)
                ]
                converted_region["coordinate_block_original"] = coord_block
                converted_regions.append(converted_region)
                
                logger.info(f"Converted region '{region.get('label', 'unnamed')}': "
                           f"{coord_block} -> [{x1_pdf:.1f}, {y1_pdf:.1f}, {x2_pdf:.1f}, {y2_pdf:.1f}]")
        
        # Create converted vision data
        converted_vision = vision_data.copy()
        converted_vision["regions"] = converted_regions
        converted_vision["conversion_info"] = {
            "scale_x": round(scale_x, 6),
            "scale_y": round(scale_y, 6),
            "pdf_dimensions": pdf_dimensions,
            "conversion_applied": True
        }
        
        return converted_vision
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return vision_data

@app.post("/process/")
async def process_pdf(file: UploadFile, vision_output: str = Form(None)):
    """Process PDF: Extract vectors, convert vision coordinates, then send to Pre-Filter API"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Parse vision_output
        vision_data = None
        if vision_output:
            try:
                vision_data = json.loads(vision_output)
                logger.info(f"Vision output parsed successfully - found {len(vision_data.get('regions', []))} regions")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for vision_output: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")

        # Call Vector Drawing API with minify=true
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'true',  # As requested
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
        
        # Parse vector response
        raw_response = vector_response.text
        logger.info(f"Raw Vector Drawing API response (first 100 chars): {raw_response[:100]}")
        logger.info(f"Response length: {len(raw_response)} bytes")
        
        try:
            vector_data = json.loads(raw_response)
            logger.info(f"JSON parsed successfully, type={type(vector_data)}")
            
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
        
        # Validate vector data structure
        if not isinstance(vector_data, dict):
            raise HTTPException(status_code=400, detail="Parsed result is not a dictionary")
            
        if 'pages' not in vector_data or not vector_data['pages']:
            raise HTTPException(status_code=400, detail="No pages found in vector data")
        
        # NEW: Convert vision coordinates if available
        converted_vision = None
        if vision_data:
            # Get PDF dimensions from vector response (coordinate_bounds from enhanced vector API)
            pdf_dimensions = vector_data.get('summary', {})
            converted_vision = convert_image_coords_to_pdf(vision_data, pdf_dimensions)
            logger.info("Vision coordinates converted to PDF coordinate system")
        
        # NEW: Prepare data for Pre-Filter API (as expected by pre-filter)
        combined_data = {
            "vision_output": converted_vision if converted_vision else vision_data,
            "vector_output": vector_data,  # Raw vector data as requested
            "coordinate_bounds": vector_data.get('summary', {}).get('coordinate_bounds', {}),
            "pdf_dimensions": vector_data.get('metadata', {}).get('pdf_dimensions', {}),
            "metadata": {
                "filename": file.filename,
                "total_pages": vector_data.get('summary', {}).get('total_pages', 1),
                "total_texts": vector_data.get('summary', {}).get('total_texts', 0),
                "dimensions_found": vector_data.get('summary', {}).get('dimensions_found', 0),
                "coordinate_conversion": "applied" if converted_vision else "not_applicable"
            }
        }
        
        logger.info("Data combined for Pre-Filter API:")
        logger.info(f"- Vision regions: {len(converted_vision.get('regions', []))} (converted)" if converted_vision else "- No vision data")
        logger.info(f"- Vector pages: {len(vector_data.get('pages', []))}")
        logger.info(f"- PDF coordinate bounds: {combined_data['coordinate_bounds']}")
        
        # Call Pre-Filter API
        try:
            headers = {'Content-Type': 'application/json'}
            logger.info("Calling Pre-Filter API with vision-enhanced data")
            
            filter_response = requests.post(
                PRE_FILTER_API_URL, 
                json=combined_data, 
                headers=headers,
                timeout=300
            )
            
            logger.info(f"Pre-Filter API response status: {filter_response.status_code}")
            
            if filter_response.status_code != 200:
                logger.warning(f"Pre-Filter API error: {filter_response.status_code}")
                logger.warning(f"Error response: {filter_response.text}")
                return {
                    "status": "partial_success",
                    "message": "Vector extraction successful, but pre-filtering failed",
                    "vector_data": vector_data,
                    "vision_data": converted_vision,
                    "filter_error": filter_response.text,
                    "timestamp": "2025-07-22"
                }
            
            # Parse Pre-Filter API response
            try:
                filtered_data = filter_response.json()
                logger.info("Pre-Filter API response parsed successfully")
            except Exception as e:
                logger.error(f"Error parsing Pre-Filter API response: {e}")
                return {
                    "status": "partial_success", 
                    "message": "Vector extraction successful, but pre-filter response parsing failed",
                    "vector_data": vector_data,
                    "vision_data": converted_vision,
                    "filter_error": f"Error parsing Pre-Filter API response: {str(e)}",
                    "timestamp": "2025-07-22"
                }
            
            # Return successful results
            return {
                "status": "success",
                "message": "PDF processed successfully through both Vector Drawing API and Pre-Filter API",
                "vector_data": vector_data,  # Keep for backward compatibility
                "scale_data": filtered_data,  # Keep for backward compatibility  
                "data": filtered_data,  # New format
                "vision_data": converted_vision,
                "processing_stats": {
                    "original_texts": vector_data.get('summary', {}).get('total_texts', 0),
                    "original_vectors": (
                        vector_data.get('summary', {}).get('total_lines', 0) +
                        vector_data.get('summary', {}).get('total_rectangles', 0) +
                        vector_data.get('summary', {}).get('total_curves', 0) +
                        vector_data.get('summary', {}).get('total_polygons', 0)
                    ),
                    "regions_processed": len(converted_vision.get("regions", [])) if converted_vision else 0,
                    "coordinate_conversion": "applied" if converted_vision else "not_applicable"
                },
                "timestamp": "2025-07-22"
            }
            
        except Exception as e:
            logger.error(f"Error calling Pre-Filter API: {e}")
            return {
                "status": "partial_success",
                "message": "Vector extraction successful, but pre-filtering failed", 
                "vector_data": vector_data,
                "vision_data": converted_vision,
                "error": f"Error calling Pre-Filter API: {str(e)}",
                "timestamp": "2025-07-22"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {
        "status": "healthy", 
        "version": "1.1.0",
        "features": [
            "Vision output integration",
            "Coordinate conversion (pixels to PDF points)", 
            "Pre-Filter API integration",
            "Raw vector data passthrough"
        ]
    }

@app.get("/")
async def root():
    return {
        "title": "Enhanced Master API",
        "version": "1.1.0", 
        "description": "Processes PDF with vision output integration for Pre-Filter API",
        "workflow": [
            "1. Parse vision output (image pixel coordinates)",
            "2. Call Vector Drawing API (minify=true)", 
            "3. Convert vision coordinates to PDF points",
            "4. Send vision_output + raw vector_output to Pre-Filter API",
            "5. Return combined results"
        ],
        "apis_used": {
            "vector_drawing": VECTOR_API_URL,
            "pre_filter": PRE_FILTER_API_URL
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
