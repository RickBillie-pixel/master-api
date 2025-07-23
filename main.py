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
    description="Processes PDF by calling Vector Drawing API and Filter API, returns minified output for Scale API",
    version="2.1.0"
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
FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/filter/"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 600

def test_vector_api_connectivity():
    """Test basic connectivity to Vector API"""
    try:
        import socket
        
        host = "vector-drawning.onrender.com"
        ip = socket.gethostbyname(host)
        logger.info(f"DNS resolution: {host} -> {ip}")
        
        test_url = f"https://{host}/"
        response = requests.get(test_url, timeout=10)
        logger.info(f"Basic connectivity test: {response.status_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"Connectivity test failed: {e}")
        return False

async def call_vector_api_with_retry(file_content: bytes, filename: str, params: dict) -> requests.Response:
    """Call Vector Drawing API with robust retry logic"""
    
    logger.info(f"=== Vector API Call ===")
    logger.info(f"File: {filename} ({len(file_content)} bytes)")
    
    connectivity_ok = test_vector_api_connectivity()
    if connectivity_ok:
        logger.info("✅ Connectivity test passed")
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            
            # Wake up the service
            try:
                wake_url = "https://vector-drawning.onrender.com/"
                wake_response = requests.get(wake_url, timeout=30)
                logger.info(f"Wake response: {wake_response.status_code}")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Wake up failed: {e}")
            
            files = {'file': (filename, file_content, 'application/pdf')}
            headers = {
                'User-Agent': 'Master-API/2.1.0',
                'Accept': 'application/json, text/plain, */*',
                'Connection': 'close'
            }
            
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
            logger.info(f"Response received after {request_time:.2f}s")
            logger.info(f"Status Code: {response.status_code}")
            
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
                    
        except requests.exceptions.Timeout as e:
            logger.warning(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=504, detail="Vector API timeout")
                
        except Exception as e:
            logger.error(f"💥 Error on attempt {attempt + 1}: {str(e)}")
            
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail=f"Vector API failed: {str(e)}")
        
        if attempt < MAX_RETRIES - 1:
            wait_time = min((2 ** attempt) + 1, 30)
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    raise HTTPException(status_code=500, detail="Vector API failed after all retries")

def convert_vision_coordinates_to_pdf(vision_data: Dict, pdf_page_size: Dict) -> Dict:
    """Convert vision coordinates from image pixels to PDF coordinates"""
    if not vision_data or not pdf_page_size:
        logger.warning("Missing vision data or PDF page size")
        return vision_data
    
    try:
        image_meta = vision_data.get("image_metadata", {})
        image_width_px = image_meta.get("image_width_pixels", 1)
        image_height_px = image_meta.get("image_height_pixels", 1)
        
        pdf_width_pts = pdf_page_size.get("width", 595)
        pdf_height_pts = pdf_page_size.get("height", 842)
        
        logger.info(f"Converting coordinates:")
        logger.info(f"  Image: {image_width_px} x {image_height_px} pixels")
        logger.info(f"  PDF: {pdf_width_pts} x {pdf_height_pts} points")
        
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
                
                # Ensure valid bounding box
                x1_pdf, x2_pdf = min(x1_pdf, x2_pdf), max(x1_pdf, x2_pdf)
                y1_pdf, y2_pdf = min(y1_pdf, y2_pdf), max(y1_pdf, y2_pdf)
                
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
                
                logger.info(f"  Region '{region.get('label', 'unnamed')}': [{x1_pdf:.1f}, {y1_pdf:.1f}, {x2_pdf:.1f}, {y2_pdf:.1f}]")
        
        converted_vision = vision_data.copy()
        converted_vision["regions"] = converted_regions
        
        logger.info(f"✅ Converted {len(converted_regions)} regions")
        return converted_vision
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return vision_data

def convert_vector_data_to_filter_format(vector_data: Dict, page_number: int = 1) -> Dict:
    """Convert Vector API output to Filter API expected format"""
    try:
        logger.info("=== Converting Vector Data to Filter Format ===")
        
        pages = vector_data.get("pages", [])
        if not pages:
            raise ValueError("No pages found in vector data")
        
        logger.info(f"Found {len(pages)} pages")
        
        converted_pages = []
        for i, page in enumerate(pages):
            logger.info(f"Converting page {i + 1}")
            
            page_size = page.get("page_size", {"width": 3370.0, "height": 2384.0})
            if not all(isinstance(page_size.get(k, 0), (int, float)) for k in ['width', 'height']):
                logger.warning(f"Invalid page_size, using default")
                page_size = {"width": 3370.0, "height": 2384.0}
            
            drawings = page.get("drawings", {})
            texts = page.get("texts", [])
            
            # Convert lines
            lines = []
            vector_lines = drawings.get("lines", [])
            for line in vector_lines:
                try:
                    start = line.get("start", [0.0, 0.0])
                    end = line.get("end", [0.0, 0.0])
                    
                    if not (isinstance(start, list) and len(start) == 2 and 
                           isinstance(end, list) and len(end) == 2):
                        continue
                    
                    if not all(isinstance(x, (int, float)) for x in start + end):
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
                    logger.warning(f"Error converting line: {e}")
                    continue
            
            logger.info(f"  Converted {len(lines)} lines")
            
            # Convert texts
            converted_texts = []
            for text in texts:
                try:
                    text_content = text.get("text", "")
                    if not text_content or not isinstance(text_content, str):
                        continue
                    
                    position = text.get("position")
                    if isinstance(position, dict):
                        x = position.get("x", 0.0)
                        y = position.get("y", 0.0)
                        if not all(isinstance(v, (int, float)) for v in [x, y]):
                            continue
                        position = [float(x), float(y)]
                    elif isinstance(position, list) and len(position) == 2:
                        if not all(isinstance(v, (int, float)) for v in position):
                            continue
                        position = [float(position[0]), float(position[1])]
                    else:
                        bbox = text.get("bounding_box", [])
                        if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                            position = [float(bbox[0]), float(bbox[1])]
                        else:
                            continue
                    
                    bbox = text.get("bounding_box", [])
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        font_size = float(text.get("font_size", 12.0))
                        text_width = len(text_content) * font_size * 0.6
                        text_height = font_size * 1.2
                        bbox = [
                            float(position[0]),
                            float(position[1]),
                            float(position[0]) + text_width,
                            float(position[1]) + text_height
                        ]
                    
                    if not all(isinstance(v, (int, float)) for v in bbox):
                        continue
                    
                    bbox = [float(v) for v in bbox]
                    x0, y0, x1, y1 = bbox
                    x0, x1 = min(x0, x1), max(x0, x1)
                    y0, y1 = min(y0, y1), max(y0, y1)
                    
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
                    logger.warning(f"Error converting text: {e}")
                    continue
            
            logger.info(f"  Converted {len(converted_texts)} texts")
            
            converted_page = {
                "page_size": page_size,
                "lines": lines,
                "texts": converted_texts,
                "symbols": []  # Skipped
            }
            converted_pages.append(converted_page)
        
        filter_vector_data = {
            "page_number": page_number,
            "pages": converted_pages
        }
        
        total_lines = sum(len(p['lines']) for p in converted_pages)
        total_texts = sum(len(p['texts']) for p in converted_pages)
        
        logger.info("✅ Vector data conversion completed")
        logger.info(f"  Total lines: {total_lines}")
        logger.info(f"  Total texts: {total_texts}")
        
        return filter_vector_data
        
    except Exception as e:
        logger.error(f"Error converting vector data: {e}")
        raise ValueError(f"Failed to convert vector data: {str(e)}")

def create_scale_api_format(filtered_data: Dict) -> Dict:
    """Convert Filter API output to Scale API input format"""
    try:
        logger.info("=== Converting to Scale API Format ===")
        
        # Collect all lines and texts from regions and unassigned
        all_lines = []
        all_texts = []
        
        # Process regions
        for region in filtered_data.get("regions", []):
            region_label = region.get("label", "")
            
            # Add region lines
            for line in region.get("vector_data", []):
                scale_line = {
                    "type": "line",
                    "p1": {
                        "x": line["p1"]["x"],
                        "y": line["p1"]["y"]
                    },
                    "p2": {
                        "x": line["p2"]["x"], 
                        "y": line["p2"]["y"]
                    },
                    "length": line["length"],
                    "orientation": line["orientation"],
                    "region": region_label
                }
                all_lines.append(scale_line)
            
            # Add region texts
            for text in region.get("texts", []):
                scale_text = {
                    "text": text["text"],
                    "position": {
                        "x": text["position"]["x"],
                        "y": text["position"]["y"]
                    },
                    "region": region_label
                }
                all_texts.append(scale_text)
        
        # Process unassigned elements
        for line in filtered_data.get("unassigned_vector_data", []):
            scale_line = {
                "type": "line",
                "p1": {
                    "x": line["p1"]["x"],
                    "y": line["p1"]["y"]
                },
                "p2": {
                    "x": line["p2"]["x"],
                    "y": line["p2"]["y"]
                },
                "length": line["length"],
                "orientation": line["orientation"],
                "region": "unassigned"
            }
            all_lines.append(scale_line)
        
        for text in filtered_data.get("unassigned_texts", []):
            scale_text = {
                "text": text["text"],
                "position": {
                    "x": text["position"]["x"],
                    "y": text["position"]["y"]
                },
                "region": "unassigned"
            }
            all_texts.append(scale_text)
        
        # Create Scale API compatible format
        scale_api_data = {
            "vector_data": all_lines,
            "texts": all_texts
        }
        
        logger.info(f"✅ Scale API format created:")
        logger.info(f"  Lines: {len(all_lines)}")
        logger.info(f"  Texts: {len(all_texts)}")
        
        return scale_api_data
        
    except Exception as e:
        logger.error(f"Error converting to Scale API format: {e}")
        raise ValueError(f"Failed to convert to Scale API format: {str(e)}")

def create_summary_output(result: Dict) -> str:
    """Create a compact summary output"""
    lines = []
    lines.append("=== MASTER API PROCESSING RESULT ===")
    lines.append(f"Status: {result.get('status', 'unknown')}")
    lines.append(f"Timestamp: {result.get('timestamp', 'unknown')}")
    
    if 'filtered_data' in result:
        filtered = result['filtered_data']
        lines.append(f"Drawing Type: {filtered.get('drawing_type', 'unknown')}")
        lines.append(f"Page: {filtered.get('page_number', 1)}")
        
        # Region summary
        regions = filtered.get('regions', [])
        lines.append(f"Regions: {len(regions)}")
        
        total_region_lines = 0
        total_region_texts = 0
        
        for region in regions:
            region_lines = len(region.get('vector_data', []))
            region_texts = len(region.get('texts', []))
            total_region_lines += region_lines
            total_region_texts += region_texts
            lines.append(f"  - {region.get('label', 'unnamed')}: {region_lines} lines, {region_texts} texts")
        
        # Unassigned summary
        unassigned_lines = len(filtered.get('unassigned_vector_data', []))
        unassigned_texts = len(filtered.get('unassigned_texts', []))
        lines.append(f"Unassigned: {unassigned_lines} lines, {unassigned_texts} texts")
        
        # Totals
        total_lines = total_region_lines + unassigned_lines
        total_texts = total_region_texts + unassigned_texts
        lines.append(f"TOTAL: {total_lines} lines, {total_texts} texts")
        
        # Metadata
        if 'metadata' in filtered:
            metadata = filtered['metadata']
            lines.append(f"Processing time: {metadata.get('processing_time_seconds', 0):.3f}s")
    
    if 'scale_api_data' in result:
        scale_data = result['scale_api_data']
        lines.append(f"Scale API Ready: {len(scale_data.get('vector_data', []))} lines, {len(scale_data.get('texts', []))} texts")
    
    return "\n".join(lines)

@app.post("/process/")
async def process_pdf(
    file: UploadFile = File(...),
    vision_output: str = Form(...),
    output_format: str = Form(default="json")
):
    """Process PDF workflow with minified output optimized for Scale API"""
    try:
        logger.info(f"=== Starting PDF Processing v2.1.0 ===")
        logger.info(f"File: {file.filename}, Output: {output_format}")
        
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
            logger.info(f"Vision output parsed - {regions_count} regions found")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")

        # Call Vector API
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
            logger.info(f"Vector API response: {len(raw_response)} bytes")
            
            vector_data = json.loads(raw_response)
            
            if isinstance(vector_data, str):
                vector_data = json.loads(vector_data)
            
            if not isinstance(vector_data, dict) or 'pages' not in vector_data:
                raise ValueError("Invalid vector data structure")
                
        except Exception as e:
            logger.error(f"Vector API parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector API response: {str(e)}")

        # Extract PDF page dimensions
        logger.info("=== Processing Coordinates ===")
        
        first_page = vector_data['pages'][0]
        pdf_page_size = first_page.get('page_size', {"width": 3370.0, "height": 2384.0})
        logger.info(f"PDF page size: {pdf_page_size}")

        # Convert vision coordinates
        converted_vision = convert_vision_coordinates_to_pdf(vision_data, pdf_page_size)

        # Convert vector data
        logger.info("=== Converting Vector Data ===")
        filter_vector_data = convert_vector_data_to_filter_format(vector_data, page_number=1)

        # Prepare Filter API request
        filter_request_data = {
            "vector_data": filter_vector_data,
            "vision_output": converted_vision
        }
        
        logger.info(f"Filter API request prepared:")
        logger.info(f"  Drawing type: {converted_vision.get('drawing_type')}")
        logger.info(f"  Regions: {len(converted_vision.get('regions', []))}")

        # Call Filter API
        logger.info("=== Calling Filter API ===")
        
        try:
            headers = {'Content-Type': 'application/json'}
            
            filter_response = requests.post(
                FILTER_API_URL,
                json=filter_request_data,
                headers=headers,
                timeout=300
            )
            
            logger.info(f"Filter API response: {filter_response.status_code}")
            
            if filter_response.status_code != 200:
                logger.error(f"Filter API error: {filter_response.text}")
                
                result = {
                    "status": "partial_success",
                    "message": "Vector extraction successful, but filtering failed",
                    "filter_error": filter_response.text,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                filtered_data = filter_response.json()
                logger.info("✅ Filter API successful")
                
                # Convert to Scale API format
                logger.info("=== Creating Scale API Format ===")
                scale_api_data = create_scale_api_format(filtered_data)
                
                result = {
                    "status": "success",
                    "message": "PDF processed successfully - optimized for Scale API",
                    "filtered_data": filtered_data,
                    "scale_api_data": scale_api_data,
                    "processing_stats": {
                        "pdf_page_size": pdf_page_size,
                        "regions_converted": len(converted_vision.get('regions', [])),
                        "coordinate_conversion": "applied"
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # Return appropriate format
            if output_format == "txt":
                summary_output = create_summary_output(result)
                return PlainTextResponse(content=summary_output)
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error calling Filter API: {e}")
            result = {
                "status": "partial_success",
                "message": "Vector extraction successful, but filtering failed",
                "error": f"Filter API error: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if output_format == "txt":
                summary_output = create_summary_output(result)
                return PlainTextResponse(content=summary_output)
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
        "version": "2.1.0",
        "description": "Master API optimized for Scale API workflow"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Master API",
        "version": "2.1.0",
        "description": "Processes PDF and returns minified output optimized for Scale API",
        "workflow": [
            "1. Receive PDF and vision_output",
            "2. Call Vector Drawing API",
            "3. Convert vision coordinates to PDF coordinates", 
            "4. Convert vector data to Filter API format",
            "5. Call Filter API (returns minified output)",
            "6. Convert to Scale API format",
            "7. Return both filtered data and Scale API ready format"
        ],
        "optimizations": [
            "Minified Filter API output",
            "Scale API compatible format",
            "Line midpoints calculated",
            "Region-based filtering",
            "No symbols (performance)"
        ],
        "output_formats": {
            "json": "Full structured data",
            "txt": "Summary text output"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Master API v2.1.0 on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
