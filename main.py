# main.py (Master API)
import os
import tempfile
import uuid
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
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API with correct JSON structure",
    version="1.5.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API URLs
VECTOR_API_URL = "https://vector-drawning.onrender.com/extract/"
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/filter/"

# Retry settings
MAX_RETRIES = 3
REQUEST_TIMEOUT = 600

def test_vector_api_connectivity():
    """Test basic connectivity to Vector API"""
    try:
        import socket
        
        # Test DNS resolution
        host = "vector-drawning.onrender.com"
        ip = socket.gethostbyname(host)
        logger.info(f"DNS resolution: {host} -> {ip}")
        
        # Test basic HTTP connectivity
        test_url = f"https://{host}/"
        response = requests.get(test_url, timeout=10)
        logger.info(f"Basic connectivity test: {response.status_code}")
        
        # Test health endpoint
        health_url = f"https://{host}/health/"
        health_response = requests.get(health_url, timeout=10)
        logger.info(f"Health endpoint test: {health_response.status_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"Connectivity test failed: {e}")
        return False

async def call_vector_api_with_retry(file_content: bytes, filename: str, params: dict) -> requests.Response:
    """Call Vector Drawing API with robust retry logic"""
    
    logger.info(f"=== Vector API Call Details ===")
    logger.info(f"URL: {VECTOR_API_URL}")
    logger.info(f"File: {filename} ({len(file_content)} bytes)")
    logger.info(f"Params: {params}")
    
    # Test connectivity first
    connectivity_ok = test_vector_api_connectivity()
    if not connectivity_ok:
        logger.error("❌ Basic connectivity test failed!")
    else:
        logger.info("✅ Basic connectivity test passed")
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"=== Vector API Attempt {attempt + 1}/{MAX_RETRIES} ===")
            
            # Wake up the service with a simple GET first
            try:
                wake_url = "https://vector-drawning.onrender.com/"
                logger.info(f"🔄 Waking up Vector API service...")
                wake_response = requests.get(wake_url, timeout=30)
                logger.info(f"Wake response: {wake_response.status_code}")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Wake up request failed: {e}")
            
            # Prepare request
            files = {'file': (filename, file_content, 'application/pdf')}
            headers = {
                'User-Agent': 'Master-API/1.5.1',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'close'
            }
            
            logger.info(f"📤 Making POST request to {VECTOR_API_URL}")
            
            start_time = time.time()
            
            response = requests.post(
                VECTOR_API_URL,
                files=files,
                params=params,
                headers=headers,
                timeout=(30, 600),
                stream=False,
                verify=True
            )
            
            request_time = time.time() - start_time
            logger.info(f"📥 Response received after {request_time:.2f}s")
            
            logger.info(f"📊 Vector API Response:")
            logger.info(f"  Status Code: {response.status_code}")
            logger.info(f"  Content Length: {len(response.content)}")
            
            if response.status_code == 200:
                logger.info("✅ Vector API call successful")
                return response
            else:
                logger.error(f"❌ Vector API returned {response.status_code}: {response.text}")
                
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Vector API failed after {MAX_RETRIES} attempts. Status: {response.status_code}"
                    )
                    
        except requests.exceptions.Timeout as e:
            logger.warning(f"⏰ Timeout on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=504,
                    detail=f"Vector API timeout after {MAX_RETRIES} attempts"
                )
                
        except Exception as e:
            logger.error(f"💥 Error on attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
            
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Vector API failed after {MAX_RETRIES} attempts: {str(e)}"
                )
        
        # Wait before retry
        if attempt < MAX_RETRIES - 1:
            wait_time = min((2 ** attempt) + 1, 30)
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    raise HTTPException(status_code=500, detail="Vector API failed after all retries")

def convert_vision_coordinates_to_pdf(vision_data: Dict, pdf_page_size: Dict) -> Dict:
    """Convert vision coordinates from image pixels to PDF coordinates"""
    if not vision_data or not pdf_page_size:
        logger.warning("Missing vision data or PDF page size for coordinate conversion")
        return vision_data
    
    try:
        # Get image metadata
        image_meta = vision_data.get("image_metadata", {})
        image_width_px = image_meta.get("image_width_pixels", 1)
        image_height_px = image_meta.get("image_height_pixels", 1)
        
        # Get PDF page dimensions
        pdf_width_pts = pdf_page_size.get("width", 595)
        pdf_height_pts = pdf_page_size.get("height", 842)
        
        logger.info(f"Converting coordinates:")
        logger.info(f"  Image: {image_width_px} x {image_height_px} pixels")
        logger.info(f"  PDF: {pdf_width_pts} x {pdf_height_pts} points")
        
        # Calculate scale factors
        scale_x = pdf_width_pts / image_width_px
        scale_y = pdf_height_pts / image_height_px
        
        # Convert regions to PDF coordinates
        converted_regions = []
        for region in vision_data.get("regions", []):
            coord_block = region.get("coordinate_block", [])
            if len(coord_block) >= 4:
                x1_img, y1_img, x2_img, y2_img = coord_block
                
                # Convert to PDF points
                x1_pdf = x1_img * scale_x
                y1_pdf = y1_img * scale_y
                x2_pdf = x2_img * scale_x
                y2_pdf = y2_img * scale_y
                
                # Ensure correct order (x1 < x2, y1 < y2)
                x1_pdf, x2_pdf = min(x1_pdf, x2_pdf), max(x1_pdf, x2_pdf)
                y1_pdf, y2_pdf = min(y1_pdf, y2_pdf), max(y1_pdf, y2_pdf)
                
                # Add small delta if zero area
                if x1_pdf == x2_pdf:
                    x2_pdf += 1.0
                if y1_pdf == y2_pdf:
                    y2_pdf += 1.0
                
                converted_region = region.copy()
                converted_region["coordinate_block"] = [
                    round(x1_pdf, 2), 
                    round(y1_pdf, 2), 
                    round(x2_pdf, 2), 
                    round(y2_pdf, 2)
                ]
                converted_regions.append(converted_region)
                
                logger.info(f"  Region '{region.get('label', 'unnamed')}':")
                logger.info(f"    PDF: [{x1_pdf:.1f}, {y1_pdf:.1f}, {x2_pdf:.1f}, {y2_pdf:.1f}]")
        
        # Return converted vision data
        converted_vision = vision_data.copy()
        converted_vision["regions"] = converted_regions
        
        logger.info(f"Successfully converted {len(converted_regions)} regions")
        return converted_vision
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return vision_data

def convert_vector_data_to_filter_format(vector_data: Dict, page_number: int = 1) -> Dict:
    """
    Convert Vector API output to Filter API expected format
    
    Vector API returns: {"pages": [...], "summary": {...}, "metadata": {...}}
    Filter API expects: {"page_number": int, "pages": [{"page_size": {...}, "lines": [...], "texts": [...], "symbols": [...]}]}
    """
    try:
        logger.info("=== Converting Vector Data to Filter Format ===")
        
        # Extract pages from Vector API response
        pages = vector_data.get("pages", [])
        if not pages:
            logger.error("No pages found in vector data")
            raise ValueError("No pages found in vector data")
        
        logger.info(f"Found {len(pages)} pages in vector data")
        
        # Convert each page to Filter API format
        converted_pages = []
        for i, page in enumerate(pages):
            logger.info(f"Converting page {i + 1}")
            
            # Get page size
            page_size = page.get("page_size", {"width": 3370.0, "height": 2384.0})
            if not all(isinstance(page_size.get(k, 0), (int, float)) for k in ['width', 'height']):
                logger.warning(f"Invalid page_size for page {i + 1}, using default")
                page_size = {"width": 3370.0, "height": 2384.0}
            logger.info(f"  Page size: {page_size}")
            
            # Extract drawings data and convert to lines/texts/symbols
            drawings = page.get("drawings", {})
            texts = page.get("texts", [])
            
            # Convert vector lines
            lines = []
            vector_lines = drawings.get("lines", [])
            for line in vector_lines:
                try:
                    # Ensure proper structure for line points
                    start = line.get("start", [0.0, 0.0])
                    end = line.get("end", [0.0, 0.0])
                    
                    # Validate points are lists with 2 elements
                    if not isinstance(start, list) or len(start) != 2:
                        logger.warning(f"Invalid start point: {start}, skipping line")
                        continue
                    if not isinstance(end, list) or len(end) != 2:
                        logger.warning(f"Invalid end point: {end}, skipping line")
                        continue
                    
                    # Ensure numeric values
                    if not all(isinstance(x, (int, float)) for x in start + end):
                        logger.warning(f"Non-numeric coordinates in line: start={start}, end={end}, skipping")
                        continue
                    
                    converted_line = {
                        "p1": [float(start[0]), float(start[1])],
                        "p2": [float(end[0]), float(end[1])],
                        "stroke_width": float(line.get("stroke_width", 1.0)),
                        "length": float(line.get("length", 0.0)),
                        "color": line.get("color", [0, 0, 0]),
                        "is_dashed": bool(line.get("is_dashed", False)),
                        "angle": float(line.get("angle")) if line.get("angle") is not None else None
                    }
                    lines.append(converted_line)
                except Exception as e:
                    logger.warning(f"Error converting line: {e}, data: {line}")
                    continue
            
            logger.info(f"  Converted {len(lines)} lines")
            
            # Convert texts
            converted_texts = []
            for text in texts:
                try:
                    # Get text content
                    text_content = text.get("text", "")
                    if not text_content or not isinstance(text_content, str):
                        logger.warning(f"Invalid or empty text content: {text_content}, skipping")
                        continue
                    
                    # Get position - handle both dict and list format
                    position = text.get("position")
                    if isinstance(position, dict):
                        x = position.get("x", 0.0)
                        y = position.get("y", 0.0)
                        if not all(isinstance(v, (int, float)) for v in [x, y]):
                            logger.warning(f"Non-numeric position values: {position}, skipping")
                            continue
                        position = [float(x), float(y)]
                    elif isinstance(position, list) and len(position) == 2:
                        if not all(isinstance(v, (int, float)) for v in position):
                            logger.warning(f"Non-numeric position values: {position}, skipping")
                            continue
                        position = [float(position[0]), float(position[1])]
                    else:
                        # Fallback: derive from bounding box
                        bbox = text.get("bounding_box", [])
                        if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                            position = [float(bbox[0]), float(bbox[1])]
                            logger.debug(f"Derived position from bbox for text '{text_content}'")
                        else:
                            logger.warning(f"Invalid position and no valid bbox for text '{text_content}': {position}, {bbox}, skipping")
                            continue
                    
                    # Get bounding box
                    bbox = text.get("bounding_box", [])
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        # Create default bounding box based on position
                        font_size = float(text.get("font_size", 12.0))
                        text_width = len(text_content) * font_size * 0.6  # Rough estimate
                        text_height = font_size * 1.2
                        bbox = [
                            float(position[0]),
                            float(position[1]),
                            float(position[0]) + text_width,
                            float(position[1]) + text_height
                        ]
                        logger.debug(f"Created default bbox for text '{text_content}': {bbox}")
                    
                    # Ensure numeric values
                    if not all(isinstance(v, (int, float)) for v in bbox):
                        logger.warning(f"Non-numeric bbox values for text '{text_content}': {bbox}, skipping")
                        continue
                    
                    # Convert to floats and ensure correct order
                    bbox = [float(v) for v in bbox]
                    x0, y0, x1, y1 = bbox
                    x0, x1 = min(x0, x1), max(x0, x1)
                    y0, y1 = min(y0, y1), max(y0, y1)
                    
                    # Add small delta if zero area
                    if x0 == x1:
                        x1 += 1.0
                    if y0 == y1:
                        y1 += 1.0
                    bbox = [x0, y0, x1, y1]
                    
                    converted_text = {
                        "text": str(text_content),
                        "position": position,
                        "font_size": float(text.get("font_size", 12.0)),
                        "bounding_box": bbox
                    }
                    converted_texts.append(converted_text)
                    
                except Exception as e:
                    logger.warning(f"Error converting text: {e}, text data: {text}")
                    continue
            
            logger.info(f"  Converted {len(converted_texts)} texts")
            
            # Convert symbols (from other drawing elements)
            symbols = []
            for shape_type in ["rectangles", "curves", "polygons", "circles"]:
                shapes = drawings.get(shape_type, [])
                for shape in shapes:
                    try:
                        bbox = shape.get("bounding_box", [])
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            # Create default bounding box from points if available
                            points = shape.get("points", [])
                            if points and all(isinstance(p, list) and len(p) == 2 for p in points):
                                xs = [float(p[0]) for p in points]
                                ys = [float(p[1]) for p in points]
                                x0, x1 = min(xs), max(xs)
                                y0, y1 = min(ys), max(ys)
                                if x0 == x1:
                                    x1 += 1.0
                                if y0 == y1:
                                    y1 += 1.0
                                bbox = [x0, y0, x1, y1]
                                logger.debug(f"Created default bbox for {shape_type} from points: {bbox}")
                            else:
                                logger.warning(f"Invalid bbox and no valid points for {shape_type}: {bbox}, {points}, skipping")
                                continue
                        
                        # Ensure numeric values
                        if not all(isinstance(v, (int, float)) for v in bbox):
                            logger.warning(f"Non-numeric bbox values for {shape_type}: {bbox}, skipping")
                            continue
                        
                        # Convert to floats and ensure correct order
                        bbox = [float(v) for v in bbox]
                        x0, y0, x1, y1 = bbox
                        x0, x1 = min(x0, x1), max(x0, x1)
                        y0, y1 = min(y0, y1), max(y0, y1)
                        
                        # Add small delta if zero area
                        if x0 == x1:
                            x1 += 1.0
                        if y0 == y1:
                            y1 += 1.0
                        bbox = [x0, y0, x1, y1]
                        
                        symbol = {
                            "type": shape_type.rstrip('s'),  # "rectangles" -> "rectangle"
                            "bounding_box": bbox,
                            "points": shape.get("points", [])
                        }
                        symbols.append(symbol)
                    except Exception as e:
                        logger.warning(f"Error converting symbol: {e}, shape data: {shape}")
                        continue
            
            logger.info(f"  Converted {len(symbols)} symbols")
            
            # Create converted page
            converted_page = {
                "page_size": page_size,
                "lines": lines,
                "texts": converted_texts,
                "symbols": symbols
            }
            converted_pages.append(converted_page)
        
        # Create final structure for Filter API
        filter_vector_data = {
            "page_number": page_number,
            "pages": converted_pages
        }
        
        logger.info("✅ Vector data conversion completed")
        logger.info(f"  Total pages: {len(converted_pages)}")
        logger.info(f"  Total lines: {sum(len(p['lines']) for p in converted_pages)}")
        logger.info(f"  Total texts: {sum(len(p['texts']) for p in converted_pages)}")
        logger.info(f"  Total symbols: {sum(len(p['symbols']) for p in converted_pages)}")
        
        return filter_vector_data
        
    except Exception as e:
        logger.error(f"Error converting vector data: {e}")
        raise ValueError(f"Failed to convert vector data: {str(e)}")

def create_minified_output(result: Dict) -> str:
    """Create a minified text output from the processing result"""
    output_lines = []
    output_lines.append("=== MASTER API PROCESSING RESULT ===\n")
    
    # Basic info
    output_lines.append(f"Status: {result.get('status', 'unknown')}")
    output_lines.append(f"Message: {result.get('message', '')}")
    output_lines.append(f"Timestamp: {result.get('timestamp', '')}\n")
    
    # Processing stats
    if 'processing_stats' in result:
        stats = result['processing_stats']
        output_lines.append("Processing Stats:")
        output_lines.append(f"  PDF Page Size: {stats.get('pdf_page_size', {})}")
        output_lines.append(f"  Regions Converted: {stats.get('regions_converted', 0)}")
        output_lines.append(f"  Version: {stats.get('version', '')}\n")
    
    # Filtered data summary
    if 'filtered_data' in result and 'filtered' in result['filtered_data']:
        filtered = result['filtered_data']['filtered']
        output_lines.append("Filtered Data Summary:")
        output_lines.append(f"  Drawing Type: {filtered.get('drawing_type', '')}")
        output_lines.append(f"  Page Number: {filtered.get('page_number', '')}")
        output_lines.append(f"  Total Regions: {len(filtered.get('regions', []))}")
        
        # Region summaries
        for region in filtered.get('regions', []):
            output_lines.append(f"\n  Region: {region.get('label', 'unnamed')}")
            output_lines.append(f"    Lines: {len(region.get('lines', []))}")
            output_lines.append(f"    Texts: {len(region.get('texts', []))}")
            output_lines.append(f"    Symbols: {len(region.get('symbols', []))}")
            
            # Sample texts
            texts = region.get('texts', [])
            if texts:
                output_lines.append("    Sample texts:")
                for text in texts[:5]:  # First 5 texts
                    output_lines.append(f"      - {text.get('text', '')}")
        
        # Unassigned elements
        unassigned = filtered.get('unassigned', {})
        output_lines.append(f"\n  Unassigned Elements:")
        output_lines.append(f"    Lines: {len(unassigned.get('lines', []))}")
        output_lines.append(f"    Texts: {len(unassigned.get('texts', []))}")
        output_lines.append(f"    Symbols: {len(unassigned.get('symbols', []))}")
    
    # Metadata
    if 'metadata' in result.get('filtered_data', {}):
        metadata = result['filtered_data']['metadata']
        output_lines.append(f"\nProcessing Metadata:")
        output_lines.append(f"  Processed Elements: {metadata.get('processed_elements', 0)}")
        output_lines.append(f"  Filtered Elements: {metadata.get('filtered_elements', 0)}")
        output_lines.append(f"  Processing Time: {metadata.get('processing_time_seconds', 0):.2f}s")
    
    # Errors
    if 'errors' in result:
        output_lines.append(f"\nErrors: {result['errors']}")
    
    return "\n".join(output_lines)

@app.post("/process/")
async def process_pdf(
    file: UploadFile = File(...),
    vision_output: str = Form(...),
    output_format: str = Form(default="json")  # json or txt
):
    """
    Process PDF workflow with correct JSON structure for Filter API
    """
    try:
        logger.info(f"=== Starting PDF Processing v1.5.1 ===")
        logger.info(f"Received file: {file.filename}")
        logger.info(f"Output format: {output_format}")
        
        # Step 1: Save uploaded PDF
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Step 2: Parse vision_output
        try:
            vision_data = json.loads(vision_output)
            regions_count = len(vision_data.get('regions', []))
            logger.info(f"Vision output parsed - {regions_count} regions found")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")

        # Step 3: Call Vector Drawing API
        logger.info("=== Calling Vector Drawing API ===")
        
        params = {
            'minify': 'true',
            'remove_non_essential': 'false',
            'precision': '2'
        }
        
        vector_response = await call_vector_api_with_retry(file_content, file.filename, params)
        
        # Parse vector response
        try:
            raw_response = vector_response.text
            logger.info(f"Vector API response length: {len(raw_response)} bytes")
            
            vector_data = json.loads(raw_response)
            
            # Handle double-encoded JSON
            if isinstance(vector_data, str):
                vector_data = json.loads(vector_data)
            
            if not isinstance(vector_data, dict) or 'pages' not in vector_data:
                raise ValueError("Invalid vector data structure")
                
        except Exception as e:
            logger.error(f"Vector API JSON parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector API response: {str(e)}")

        # Step 4: Extract PDF page dimensions
        logger.info("=== Extracting PDF Page Dimensions ===")
        
        first_page = vector_data['pages'][0]
        pdf_page_size = first_page.get('page_size', {"width": 3370.0, "height": 2384.0})
        logger.info(f"PDF page size: {pdf_page_size}")

        # Step 5: Convert vision coordinates to PDF coordinates
        logger.info("=== Converting Vision Coordinates ===")
        converted_vision = convert_vision_coordinates_to_pdf(vision_data, pdf_page_size)

        # Step 6: Convert Vector data to Filter API format
        logger.info("=== Converting Vector Data Format ===")
        filter_vector_data = convert_vector_data_to_filter_format(vector_data, page_number=1)

        # Step 7: Prepare correct JSON structure for Filter API
        logger.info("=== Preparing Filter API Request ===")
        
        filter_request_data = {
            "vector_data": filter_vector_data,
            "vision_output": converted_vision
        }
        
        logger.info("Filter API request structure:")
        logger.info(f"  vector_data.page_number: {filter_vector_data.get('page_number')}")
        logger.info(f"  vector_data.pages: {len(filter_vector_data.get('pages', []))} pages")
        logger.info(f"  vision_output.drawing_type: {converted_vision.get('drawing_type')}")
        logger.info(f"  vision_output.regions: {len(converted_vision.get('regions', []))} regions")

        # Step 8: Call Pre-Filter API with correct structure
        logger.info("=== Calling Pre-Filter API ===")
        
        try:
            headers = {'Content-Type': 'application/json'}
            
            filter_response = requests.post(
                PRE_FILTER_API_URL,
                json=filter_request_data,
                headers=headers,
                timeout=300
            )
            
            logger.info(f"Pre-Filter API response status: {filter_response.status_code}")
            
            if filter_response.status_code != 200:
                logger.error(f"Pre-Filter API error: {filter_response.status_code}")
                logger.error(f"Error details: {filter_response.text}")
                
                result = {
                    "status": "partial_success",
                    "message": "Vector extraction successful, but filtering failed",
                    "vector_data": vector_data,
                    "vision_data": converted_vision,
                    "filter_error": filter_response.text,
                    "timestamp": time.strftime("%Y-%m-%d"),
                    "processing_stats": {
                        "pdf_page_size": pdf_page_size,
                        "regions_converted": len(converted_vision.get('regions', [])),
                        "coordinate_conversion": "applied",
                        "json_structure": "corrected_for_filter_api",
                        "version": "1.5.1"
                    }
                }
            else:
                # Parse successful response
                filtered_data = filter_response.json()
                logger.info("✅ Pre-Filter API response parsed successfully")
                
                # Step 9: Return successful results
                logger.info("=== Processing Completed Successfully ===")
                
                result = {
                    "status": "success",
                    "message": "PDF processed successfully with correct JSON structure",
                    "vector_data": vector_data,
                    "filtered_data": filtered_data,
                    "vision_data": converted_vision,
                    "processing_stats": {
                        "pdf_page_size": pdf_page_size,
                        "regions_converted": len(converted_vision.get('regions', [])),
                        "coordinate_conversion": "applied",
                        "json_structure": "corrected_for_filter_api",
                        "version": "1.5.1"
                    },
                    "timestamp": time.strftime("%Y-%m-%d")
                }
            
            # Return based on requested format
            if output_format == "txt":
                minified_output = create_minified_output(result)
                return PlainTextResponse(content=minified_output)
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error calling Pre-Filter API: {e}")
            result = {
                "status": "partial_success",
                "message": "Vector extraction successful, but filtering failed",
                "vector_data": vector_data,
                "vision_data": converted_vision,
                "error": f"Pre-Filter API error: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d"),
                "processing_stats": {
                    "pdf_page_size": pdf_page_size,
                    "regions_converted": len(converted_vision.get('regions', [])),
                    "coordinate_conversion": "applied",
                    "json_structure": "corrected_for_filter_api",
                    "version": "1.5.1"
                }
            }
            
            if output_format == "txt":
                minified_output = create_minified_output(result)
                return PlainTextResponse(content=minified_output)
            else:
                return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.5.1",
        "features": [
            "PDF upload and processing",
            "Vision output parsing",
            "Vector Drawing API integration with retry logic",
            "Coordinate conversion (image pixels -> PDF points)",
            "Fixed vector data format conversion for Filter API",
            "Handles dict-format positions (x,y) to list format",
            "Robust text position and bounding box handling",
            "Correct JSON structure for Filter API",
            "Pre-Filter API integration",
            "Minified text output option"
        ],
        "fixes": [
            "Fixed text position conversion from dict to list",
            "Repairs invalid bounding boxes for texts and symbols",
            "Handles missing position with bbox fallback",
            "Repairs zero-area bounding boxes",
            "Preserves all valid texts and symbols",
            "Detailed logging for skipped elements"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Master API",
        "version": "1.5.1",
        "description": "Processes PDF with correct JSON structure for Filter API",
        "workflow": [
            "1. Receive PDF file and vision_output",
            "2. Call Vector Drawing API for vector extraction",
            "3. Extract PDF page dimensions",
            "4. Convert vision coordinates to PDF coordinates",
            "5. Convert vector data to Filter API format with proper validation",
            "6. Send correct JSON structure to Filter API",
            "7. Return combined results (JSON or minified TXT)"
        ],
        "json_structure_fix": {
            "problem": "Master API was sending invalid text positions (dict format instead of list) and losing texts/symbols",
            "solution": "Converts {'x': val, 'y': val} to [x, y], repairs invalid bounding boxes, preserves all valid data",
            "filter_api_expects": {
                "vector_data": {
                    "page_number": 1, 
                    "pages": [{
                        "page_size": {"width": "float", "height": "float"},
                        "lines": [{
                            "p1": "[float, float]",
                            "p2": "[float, float]",
                            "stroke_width": "float",
                            "length": "float",
                            "color": "[int, int, int]",
                            "is_dashed": "bool",
                            "angle": "float or null"
                        }],
                        "texts": [{
                            "text": "string",
                            "position": "[float, float]",
                            "font_size": "float",
                            "bounding_box": "[x0, y0, x2, y3] where x0 < x2 and y0 < y3"
                        }],
                        "symbols": [{
                            "type": "string",
                            "bounding_box": "[x0, y0, x2, y3] where x0 < x2 and y0 < y3",
                            "points": "[[float, float], ...]"
                        }]
                    }]
                },
                "vision_output": {"drawing_type": "...", "regions": [...]}
            }
        },
        "apis_used": {
            "vector_drawing": VECTOR_API_URL,
            "pre_filter": PRE_FILTER_API_URL
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Master API v1.5.1 on port {PORT}")
    logger.info("New: Fixed text position conversion, bounding box repairs, and preserved all valid texts/symbols")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
