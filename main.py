import os
import tempfile
import uuid
import math
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import logging
import requests
import json
import time
import asyncio
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Clean Master API",
    description="Processes PDF and returns clean, focused output per region",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_API_URL = "https://vector-drawning.onrender.com/extract/"
FILTER_API_URL = "https://your-clean-filter-api.onrender.com/filter/"  # Update this URL

MAX_RETRIES = 3

async def call_vector_api_with_retry(file_content: bytes, filename: str, params: dict) -> requests.Response:
    """Call Vector Drawing API with retry logic"""
    
    logger.info(f"=== Vector API Call ===")
    logger.info(f"File: {filename} ({len(file_content)} bytes)")
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            
            files = {'file': (filename, file_content, 'application/pdf')}
            headers = {
                'User-Agent': 'Clean-Master-API/3.0.0',
                'Accept': 'application/json',
                'Connection': 'close'
            }
            
            response = requests.post(
                VECTOR_API_URL,
                files=files,
                params=params,
                headers=headers,
                timeout=(30, 600)
            )
            
            if response.status_code == 200:
                logger.info("✅ Vector API call successful")
                return response
            else:
                logger.error(f"❌ Vector API returned {response.status_code}")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Vector API failed after {MAX_RETRIES} attempts"
                    )
                    
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=504, detail="Vector API timeout")
                
        except Exception as e:
            logger.error(f"💥 Error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail=f"Vector API failed: {str(e)}")
        
        if attempt < MAX_RETRIES - 1:
            wait_time = 2 ** attempt + 1
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    raise HTTPException(status_code=500, detail="Vector API failed after all retries")

def convert_vision_coordinates_to_pdf(vision_data: Dict, pdf_page_size: Dict) -> Dict:
    """Convert vision coordinates from image pixels to PDF coordinates"""
    if not vision_data or not pdf_page_size:
        return vision_data
    
    try:
        image_meta = vision_data.get("image_metadata", {})
        image_width_px = image_meta.get("image_width_pixels", 1)
        image_height_px = image_meta.get("image_height_pixels", 1)
        
        pdf_width_pts = pdf_page_size.get("width", 595)
        pdf_height_pts = pdf_page_size.get("height", 842)
        
        scale_x = pdf_width_pts / image_width_px
        scale_y = pdf_height_pts / image_height_px
        
        converted_regions = []
        for region in vision_data.get("regions", []):
            coord_block = region.get("coordinate_block", [])
            if len(coord_block) >= 4:
                x1_img, y1_img, x2_img, y2_img = coord_block
                
                x1_pdf = x1_img * scale_x
                y1_pdf = y1_img * scale_y
                x2_pdf = x2_img * scale_x
                y2_pdf = y2_img * scale_y
                
                converted_region = region.copy()
                converted_region["coordinate_block"] = [
                    round(x1_pdf, 1), 
                    round(y1_pdf, 1), 
                    round(x2_pdf, 1), 
                    round(y2_pdf, 1)
                ]
                converted_regions.append(converted_region)
        
        converted_vision = vision_data.copy()
        converted_vision["regions"] = converted_regions
        
        logger.info(f"✅ Converted {len(converted_regions)} regions")
        return converted_vision
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return vision_data

def convert_vector_data_to_filter_format(vector_data: Dict) -> Dict:
    """Convert Vector API output to Filter API expected format"""
    try:
        logger.info("=== DEBUG: Vector Data Structure Analysis ===")
        
        pages = vector_data.get("pages", [])
        if not pages:
            raise ValueError("No pages found in vector data")
        
        page = pages[0]
        page_size = page.get("page_size", {"width": 595.0, "height": 842.0})
        
        # Log the actual structure to understand what we're getting
        logger.info(f"Page keys: {list(page.keys())}")
        
        # Try to find the correct structure
        texts = []
        lines = []
        
        # Check for texts in multiple possible locations
        if "texts" in page:
            texts = page["texts"]
            logger.info(f"Found {len(texts)} texts in page['texts']")
        
        # Check for drawings/lines in multiple possible locations
        drawings = page.get("drawings", {})
        if drawings:
            logger.info(f"Drawings keys: {list(drawings.keys())}")
            
            # Check for lines in drawings
            if "lines" in drawings:
                raw_lines = drawings["lines"]
                logger.info(f"Found {len(raw_lines)} lines in drawings['lines']")
                
                # Log first few lines to see structure
                for i, line in enumerate(raw_lines[:3]):
                    logger.info(f"Sample line {i}: {line}")
                
                # Convert lines based on actual structure
                for line in raw_lines:
                    try:
                        # Check different possible field names for coordinates
                        start_point = None
                        end_point = None
                        
                        # Try different field names
                        if "p1" in line and "p2" in line:
                            start_point = line["p1"]
                            end_point = line["p2"]
                        elif "start" in line and "end" in line:
                            start_point = line["start"]
                            end_point = line["end"]
                        elif "type" in line and line["type"] == "line":
                            # For your Vector API format
                            start_point = line.get("p1", [0.0, 0.0])
                            end_point = line.get("p2", [0.0, 0.0])
                        
                        # Convert points to [x, y] format
                        if start_point and end_point:
                            # Handle different point formats
                            if isinstance(start_point, dict):
                                start = [float(start_point.get("x", 0)), float(start_point.get("y", 0))]
                            elif isinstance(start_point, list) and len(start_point) >= 2:
                                start = [float(start_point[0]), float(start_point[1])]
                            else:
                                start = [0.0, 0.0]
                            
                            if isinstance(end_point, dict):
                                end = [float(end_point.get("x", 0)), float(end_point.get("y", 0))]
                            elif isinstance(end_point, list) and len(end_point) >= 2:
                                end = [float(end_point[0]), float(end_point[1])]
                            else:
                                end = [0.0, 0.0]
                            
                            # Calculate length if not provided
                            length = line.get("length", 0.0)
                            if length == 0.0:
                                length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            
                            converted_line = {
                                "p1": start,
                                "p2": end,
                                "stroke_width": float(line.get("width", line.get("stroke_width", 1.0))),
                                "length": float(length),
                                "color": line.get("color", [0, 0, 0]),
                                "is_dashed": bool(line.get("is_dashed", False)),
                                "angle": float(line.get("angle")) if line.get("angle") is not None else None
                            }
                            lines.append(converted_line)
                            
                            # Log successful conversion
                            if len(lines) <= 3:
                                logger.info(f"Converted line {len(lines)}: {start} -> {end}, length: {length}")
                        
                    except Exception as e:
                        logger.warning(f"Error converting line: {e}")
                        continue
        
        # Convert texts
        converted_texts = []
        for text in texts:
            try:
                text_content = text.get("text", "")
                if not text_content:
                    continue
                
                position = text.get("position")
                if isinstance(position, dict):
                    position = [float(position.get("x", 0)), float(position.get("y", 0))]
                elif isinstance(position, list) and len(position) == 2:
                    position = [float(position[0]), float(position[1])]
                else:
                    bbox = text.get("bbox", text.get("bounding_box", [0, 0, 100, 20]))
                    if isinstance(bbox, dict):
                        position = [float(bbox.get("x0", 0)), float(bbox.get("y0", 0))]
                    else:
                        position = [float(bbox[0]), float(bbox[1])]
                
                bbox = text.get("bbox", text.get("bounding_box", []))
                if isinstance(bbox, dict):
                    bbox = [bbox.get("x0", 0), bbox.get("y0", 0), bbox.get("x1", 100), bbox.get("y1", 20)]
                elif len(bbox) != 4:
                    font_size = float(text.get("font_size", 12.0))
                    text_width = len(text_content) * font_size * 0.6
                    text_height = font_size * 1.2
                    bbox = [
                        position[0],
                        position[1],
                        position[0] + text_width,
                        position[1] + text_height
                    ]
                
                converted_text = {
                    "text": str(text_content),
                    "position": position,
                    "font_size": float(text.get("font_size", 12.0)),
                    "bounding_box": [float(x) for x in bbox]
                }
                converted_texts.append(converted_text)
                
            except Exception as e:
                logger.warning(f"Error converting text: {e}")
                continue
        
        filter_data = {
            "page_number": 1,
            "pages": [{
                "page_size": page_size,
                "lines": lines,
                "texts": converted_texts
            }]
        }
        
        logger.info(f"✅ Converted: {len(lines)} lines, {len(converted_texts)} texts")
        
        if len(lines) == 0:
            logger.error("⚠️ NO LINES CONVERTED! Check Vector API response structure")
        elif len(lines) > 0:
            sample_line = lines[0]
            logger.info(f"Sample converted line: {sample_line['p1']} -> {sample_line['p2']}")
        
        return filter_data
        
    except Exception as e:
        logger.error(f"Error converting vector data: {e}")
        raise ValueError(f"Failed to convert vector data: {str(e)}")

@app.post("/process/")
async def process_pdf_clean(
    file: UploadFile = File(...),
    vision_output: str = Form(...),
    output_format: str = Form(default="json"),
    debug: bool = Form(default=False)
):
    """Process PDF and return clean, focused output per region"""
    try:
        logger.info(f"=== Starting Clean PDF Processing v3.0.0 ===")
        logger.info(f"File: {file.filename}, Debug: {debug}")
        
        # Read file
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Parse vision output
        try:
            vision_data = json.loads(vision_output)
            regions_count = len(vision_data.get('regions', []))
            logger.info(f"Vision regions: {regions_count}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")

        # Call Vector API
        logger.info("=== Calling Vector API ===")
        
        params = {
            'minify': 'true',
            'precision': '1'
        }
        
        vector_response = await call_vector_api_with_retry(file_content, file.filename, params)
        
        # Parse vector response
        try:
            vector_data = json.loads(vector_response.text)
            if isinstance(vector_data, str):
                vector_data = json.loads(vector_data)
                
        except Exception as e:
            logger.error(f"Vector API parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector API response")

        # Extract PDF page dimensions and convert coordinates
        logger.info("=== Converting Coordinates ===")
        
        first_page = vector_data['pages'][0]
        pdf_page_size = first_page.get('page_size', {"width": 595.0, "height": 842.0})
        converted_vision = convert_vision_coordinates_to_pdf(vision_data, pdf_page_size)

        # Convert vector data
        logger.info("=== Converting Vector Data ===")
        filter_vector_data = convert_vector_data_to_filter_format(vector_data)

        # Prepare Filter API request
        filter_request = {
            "vector_data": filter_vector_data,
            "vision_output": converted_vision
        }

        # Call Clean Filter API with debug if requested
        logger.info("=== Calling Clean Filter API ===")
        
        try:
            debug_data = None
            if debug:
                # First call debug endpoint
                debug_response = requests.post(
                    FILTER_API_URL.replace('/filter/', '/debug/'),
                    json=filter_request,
                    headers={'Content-Type': 'application/json'},
                    timeout=300
                )
                
                if debug_response.status_code == 200:
                    debug_data = debug_response.json()
                    logger.info("=== DEBUG INFO ===")
                    logger.info(f"Total lines: {debug_data.get('total_lines')}")
                    logger.info(f"Coordinate range: {debug_data.get('coordinate_analysis')}")
                    for region_debug in debug_data.get('regions', []):
                        logger.info(f"Region {region_debug['label']}: {region_debug['lines_in_region']} lines in region, {region_debug['lines_included']} included")
                else:
                    logger.warning("Debug endpoint failed")
            
            # Main filter call with debug parameter
            filter_url = f"{FILTER_API_URL}?debug={str(debug).lower()}"
            filter_response = requests.post(
                filter_url,
                json=filter_request,
                headers={'Content-Type': 'application/json'},
                timeout=300
            )
            
            if filter_response.status_code != 200:
                logger.error(f"Filter API error: {filter_response.text}")
                raise HTTPException(status_code=500, detail="Filter API failed")
            
            filtered_data = filter_response.json()
            logger.info("✅ Clean filtering successful")
            
            # Create final result
            result = {
                "status": "success",
                "data": filtered_data,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if debug and debug_data:
                result["debug_info"] = debug_data
            
            # Log summary
            total_lines = sum(len(r.get('lines', [])) for r in filtered_data.get('regions', []))
            total_texts = sum(len(r.get('texts', [])) for r in filtered_data.get('regions', []))
            
            logger.info(f"✅ Final result:")
            logger.info(f"  Drawing type: {filtered_data.get('drawing_type')}")
            logger.info(f"  Regions: {len(filtered_data.get('regions', []))}")
            logger.info(f"  Total lines: {total_lines}")
            logger.info(f"  Total texts: {total_texts}")
            
            if total_lines == 0:
                logger.warning("⚠️  WARNING: No lines in output! Check coordinate conversion or filtering logic.")
            
            # Return appropriate format
            if output_format == "txt":
                summary = f"""=== CLEAN PDF PROCESSING RESULT ===
Status: {result['status']}
Drawing Type: {filtered_data.get('drawing_type')}
Regions: {len(filtered_data.get('regions', []))}
Total Lines: {total_lines}
Total Texts: {total_texts}
Timestamp: {result['timestamp']}

REGIONS:
"""
                for region in filtered_data.get('regions', []):
                    summary += f"- {region.get('label')}: {len(region.get('lines', []))} lines, {len(region.get('texts', []))} texts\n"
                
                if total_lines == 0:
                    summary += "\n⚠️  WARNING: No lines found! Check coordinate system or filtering logic.\n"
                
                return PlainTextResponse(content=summary)
            else:
                return result
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Filter API request failed: {e}")
            raise HTTPException(status_code=500, detail="Filter API connection failed")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/debug/")
async def debug_pdf_processing(
    file: UploadFile = File(...),
    vision_output: str = Form(...)
):
    """Debug endpoint to diagnose processing issues"""
    try:
        logger.info("=== Debug PDF Processing ===")
        
        # Read file
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        # Parse vision output
        vision_data = json.loads(vision_output)
        
        # Call Vector API
        params = {'minify': 'true', 'precision': '1'}
        vector_response = await call_vector_api_with_retry(file_content, file.filename, params)
        vector_data = json.loads(vector_response.text)
        
        # Convert coordinates
        first_page = vector_data['pages'][0]
        pdf_page_size = first_page.get('page_size', {"width": 595.0, "height": 842.0})
        converted_vision = convert_vision_coordinates_to_pdf(vision_data, pdf_page_size)
        
        # Convert vector data
        filter_vector_data = convert_vector_data_to_filter_format(vector_data)
        
        # Call debug endpoint
        filter_request = {
            "vector_data": filter_vector_data,
            "vision_output": converted_vision
        }
        
        debug_response = requests.post(
            FILTER_API_URL.replace('/filter/', '/debug/'),
            json=filter_request,
            headers={'Content-Type': 'application/json'},
            timeout=300
        )
        
        debug_data = debug_response.json() if debug_response.status_code == 200 else {"error": debug_response.text}
        
        return {
            "file_info": {
                "filename": file.filename,
                "size_mb": round(file_size_mb, 2)
            },
            "pdf_page_size": pdf_page_size,
            "vision_regions": len(vision_data.get('regions', [])),
            "converted_regions": [
                {
                    "label": r.get('label'),
                    "original_bounds": vision_data['regions'][i]['coordinate_block'],
                    "converted_bounds": r.get('coordinate_block')
                }
                for i, r in enumerate(converted_vision.get('regions', []))
            ],
            "vector_data_info": {
                "total_lines": len(filter_vector_data['pages'][0]['lines']),
                "total_texts": len(filter_vector_data['pages'][0]['texts'])
            },
            "filter_debug": debug_data
        }
        
    except Exception as e:
        logger.error(f"Debug error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@app.get("/health/")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "description": "Clean Master API - focused output per region"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Clean Master API",
        "version": "3.0.0",
        "description": "Processes PDF and returns clean, focused output per region",
        "workflow": [
            "1. Receive PDF and vision_output",
            "2. Call Vector Drawing API",
            "3. Convert coordinates to PDF space",
            "4. Call Clean Filter API",
            "5. Return focused data per region only"
        ],
        "key_improvements": [
            "No unassigned data",
            "No unnecessary metadata", 
            "Clean per-region structure",
            "Precise line data with midpoints",
            "Text with bounding boxes preserved",
            "Correct length filtering per drawing type (plattegrond > 50pt)",
            "Debug mode for troubleshooting"
        ],
        "endpoints": {
            "/process/": "Main processing endpoint",
            "/debug/": "Debug coordinate conversion and filtering",
            "/process/?debug=true": "Process with debug logging"
        },
        "output_structure": {
            "status": "success",
            "data": {
                "drawing_type": "plattegrond",
                "regions": [
                    {
                        "label": "region_name",
                        "lines": "array of clean line objects",
                        "texts": "array of clean text objects"
                    }
                ]
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Clean Master API v3.0.0 on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
