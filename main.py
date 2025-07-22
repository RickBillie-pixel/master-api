import os
import tempfile
import uuid
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API with coordinate conversion",
    version="1.0.4"
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

def convert_pixel_to_pdf_coordinates(vision_output: Dict[str, Any], pdf_width: float, pdf_height: float) -> Dict[str, Any]:
    """
    Convert pixel coordinates from vision output to PDF coordinates
    
    Args:
        vision_output: Vision API output with pixel coordinates
        pdf_width: PDF page width in points
        pdf_height: PDF page height in points
    
    Returns:
        Updated vision output with PDF coordinates
    """
    try:
        # Ensure vision_output is a dictionary
        if not isinstance(vision_output, dict):
            logger.error(f"Vision output is not a dictionary: {type(vision_output)}")
            return {
                "coordinate_conversion": {
                    "conversion_applied": False,
                    "error": f"Vision output is not a dictionary but {type(vision_output)}"
                }
            }
        
        # Get image dimensions from vision output
        image_metadata = vision_output.get("image_metadata", {})
        image_width = image_metadata.get("image_width_pixels", 0)
        image_height = image_metadata.get("image_height_pixels", 0)
        
        if image_width <= 0 or image_height <= 0:
            logger.warning("Invalid image dimensions in vision output")
            return vision_output
        
        # Calculate scaling factors
        scale_x = pdf_width / image_width
        scale_y = pdf_height / image_height
        
        logger.info(f"Converting coordinates - Image: {image_width}x{image_height}, PDF: {pdf_width}x{pdf_height}")
        logger.info(f"Scale factors - X: {scale_x:.6f}, Y: {scale_y:.6f}")
        
        # Create a copy of the vision output
        converted_output = vision_output.copy()
        
        # Convert region coordinates
        if "regions" in converted_output:
            converted_regions = []
            for region in converted_output["regions"]:
                if "coordinate_block" in region and isinstance(region["coordinate_block"], list) and len(region["coordinate_block"]) >= 4:
                    # Original pixel coordinates [x1, y1, x2, y2]
                    pixel_coords = region["coordinate_block"]
                    
                    # Convert to PDF coordinates
                    pdf_x1 = round(pixel_coords[0] * scale_x, 2)
                    pdf_y1 = round(pixel_coords[1] * scale_y, 2)
                    pdf_x2 = round(pixel_coords[2] * scale_x, 2)
                    pdf_y2 = round(pixel_coords[3] * scale_y, 2)
                    
                    # Create new region with converted coordinates
                    converted_region = region.copy()
                    converted_region["coordinate_block"] = [pdf_x1, pdf_y1, pdf_x2, pdf_y2]
                    converted_region["original_pixel_coordinates"] = pixel_coords  # Keep original for reference
                    
                    converted_regions.append(converted_region)
                    
                    logger.info(f"Converted region '{region.get('label', 'Unknown')}': "
                              f"Pixels [{pixel_coords[0]}, {pixel_coords[1]}, {pixel_coords[2]}, {pixel_coords[3]}] -> "
                              f"PDF [{pdf_x1}, {pdf_y1}, {pdf_x2}, {pdf_y2}]")
                else:
                    # Keep regions without valid coordinates as-is
                    converted_regions.append(region)
                    logger.warning(f"Region '{region.get('label', 'Unknown')}' has invalid coordinate_block")
            
            converted_output["regions"] = converted_regions
        
        # Add conversion metadata
        converted_output["coordinate_conversion"] = {
            "source_image_size": {"width": image_width, "height": image_height},
            "target_pdf_size": {"width": pdf_width, "height": pdf_height},
            "scale_factors": {"x": scale_x, "y": scale_y},
            "conversion_applied": True
        }
        
        logger.info(f"Successfully converted {len(converted_output.get('regions', []))} regions")
        return converted_output
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        # Return safe fallback
        return {
            "coordinate_conversion": {
                "conversion_applied": False,
                "error": str(e)
            }
        }

@app.post("/process/")
async def process_pdf(file: UploadFile, vision_output: str = Form(...)):
    """Process PDF: Extract vectors via Vector Drawing API, convert coordinates, then filter via Pre-Filter API"""
    filtered_data = None
    vector_data = None
    
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Parse vision_output (handle potential double encoding)
        try:
            logger.info(f"Raw vision_output type: {type(vision_output)}")
            logger.info(f"Raw vision_output preview: {str(vision_output)[:200]}...")
            
            vision_output_dict = json.loads(vision_output)
            
            # Check if it's still a string (double encoded)
            if isinstance(vision_output_dict, str):
                logger.info("Vision output is double-encoded, parsing again")
                vision_output_dict = json.loads(vision_output_dict)
            
            logger.info("Vision output parsed successfully")
            logger.info(f"Vision output type after parsing: {type(vision_output_dict)}")
            
            if isinstance(vision_output_dict, dict):
                logger.info(f"Vision output contains {len(vision_output_dict.get('regions', []))} regions")
            else:
                logger.warning(f"Vision output is not a dict: {type(vision_output_dict)}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for vision_output: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing vision_output: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing vision_output: {str(e)}")
        
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
            # The vector API returns minified JSON as a string, not a JSON object
            vector_data = json.loads(raw_response)
            logger.info("Parsed Vector Drawing API response successfully")
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector Drawing API response: {str(e)}")
        
        # Validate structure
        if not isinstance(vector_data, dict):
            raise HTTPException(status_code=400, detail="Vector Drawing API response is not a dictionary")
            
        if 'pages' not in vector_data:
            raise HTTPException(status_code=400, detail="Vector Drawing API response missing 'pages' field")
        
        # Step 3: Extract PDF dimensions for coordinate conversion
        pdf_width = 0
        pdf_height = 0
        
        if vector_data.get("pages") and len(vector_data["pages"]) > 0:
            first_page = vector_data["pages"][0]
            page_size = first_page.get("page_size", {})
            pdf_width = page_size.get("width", 0)
            pdf_height = page_size.get("height", 0)
            
            logger.info(f"PDF dimensions: {pdf_width} x {pdf_height} points")
        
        if pdf_width <= 0 or pdf_height <= 0:
            logger.warning("Could not determine valid PDF dimensions, using default conversion")
            # Try from metadata if available
            metadata = vector_data.get("metadata", {})
            if "pdf_dimensions" in metadata:
                pdf_width = metadata["pdf_dimensions"].get("width", 612)  # Default letter size
                pdf_height = metadata["pdf_dimensions"].get("height", 792)
            else:
                pdf_width = 612  # Default letter size width
                pdf_height = 792  # Default letter size height
        
        # Step 4: Convert vision output coordinates to PDF coordinates
        logger.info("Converting vision output coordinates to PDF coordinates")
        if isinstance(vision_output_dict, dict):
            converted_vision_output = convert_pixel_to_pdf_coordinates(vision_output_dict, pdf_width, pdf_height)
        else:
            logger.error(f"Cannot convert coordinates: vision_output_dict is {type(vision_output_dict)}")
            converted_vision_output = {
                "coordinate_conversion": {
                    "conversion_applied": False,
                    "error": f"Vision output is not a dictionary but {type(vision_output_dict)}"
                }
            }
        
        # Step 5: Combine data with converted coordinates
        combined_data = {
            "vision_output": converted_vision_output,
            "vector_output": vector_data,
            "page_dimensions": {
                "width": pdf_width,
                "height": pdf_height
            },
            "coordinate_system": "pdf_points"
        }
        logger.info("Data combined successfully with converted coordinates")
        
        # Step 6: Send to Pre-Filter API
        logger.info("Calling Pre-Filter API with converted coordinate data")
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
            # Return partial success with converted coordinates
            return {
                "status": "partial_success",
                "message": "Vector extraction and coordinate conversion successful, but pre-filtering failed",
                "data": combined_data,
                "filter_error": filter_response.text,
                "coordinate_conversion_applied": converted_vision_output.get("coordinate_conversion", {}).get("conversion_applied", False)
            }
        
        # Parse Pre-Filter response
        try:
            filtered_data = filter_response.json()
            logger.info("Pre-Filter API response parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing Pre-Filter API response: {e}")
            return {
                "status": "partial_success",
                "message": "Vector extraction and coordinate conversion successful, but pre-filter response parsing failed",
                "data": combined_data,
                "filter_error": str(e),
                "coordinate_conversion_applied": converted_vision_output.get("coordinate_conversion", {}).get("conversion_applied", False)
            }
        
        # Return success
        return {
            "status": "success",
            "message": "PDF processed successfully through both APIs with coordinate conversion",
            "data": filtered_data,
            "coordinate_conversion_applied": converted_vision_output.get("coordinate_conversion", {}).get("conversion_applied", False),
            "conversion_details": converted_vision_output.get("coordinate_conversion", {})
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0.4"}

@app.get("/")
async def root():
    return {
        "title": "Master API",
        "description": "Processes PDFs through Vector Drawing API and Pre-Filter API with coordinate conversion",
        "version": "1.0.4",
        "features": [
            "PDF processing via Vector Drawing API",
            "Coordinate conversion from pixel to PDF coordinates", 
            "Data filtering via Pre-Filter API",
            "Automatic coordinate system alignment"
        ],
        "endpoints": {
            "/": "This page",
            "/process/": "POST - Process PDF file with vision output",
            "/health/": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
