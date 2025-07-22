import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import json
import time
import asyncio
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API with robust error handling",
    version="1.2.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API URLs - CORRECT: Using actual deployed URL
VECTOR_API_URL = "https://vector-drawning.onrender.com/extract/"  # Correct URL as confirmed
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

# Add DNS and connectivity debugging
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
        logger.info(f"Health response: {health_response.text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Connectivity test failed: {e}")
        return False

# Retry settings
MAX_RETRIES = 3
REQUEST_TIMEOUT = 600  # 10 minutes

async def call_vector_api_with_retry(file_content: bytes, filename: str, params: dict) -> requests.Response:
    """Call Vector Drawing API with robust retry logic and detailed debugging"""
    
    logger.info(f"=== Vector API Call Details ===")
    logger.info(f"URL: {VECTOR_API_URL}")
    logger.info(f"File: {filename} ({len(file_content)} bytes)")
    logger.info(f"Params: {params}")
    
    # Test connectivity first
    logger.info("=== Testing Vector API Connectivity ===")
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
                time.sleep(2)  # Give it a moment to wake up
            except Exception as e:
                logger.warning(f"Wake up request failed: {e}")
            
            # Prepare request with detailed logging
            files = {'file': (filename, file_content, 'application/pdf')}
            headers = {
                'User-Agent': 'Master-API/1.2.1',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'close'
            }
            
            logger.info(f"📤 Making POST request to {VECTOR_API_URL}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Files: file='{filename}' ({len(file_content)} bytes, type='application/pdf')")
            logger.info(f"Params: {params}")
            
            # Use requests.post directly with verbose logging
            start_time = time.time()
            
            try:
                response = requests.post(
                    VECTOR_API_URL,
                    files=files,
                    params=params,
                    headers=headers,
                    timeout=(30, 600),  # (connect_timeout, read_timeout)
                    stream=False,
                    verify=True
                )
                
                request_time = time.time() - start_time
                logger.info(f"📥 Response received after {request_time:.2f}s")
                
            except requests.exceptions.ConnectTimeout:
                logger.error("❌ Connection timeout (30s)")
                raise
            except requests.exceptions.ReadTimeout:
                logger.error("❌ Read timeout (600s)")
                raise
            except requests.exceptions.ConnectionError as e:
                logger.error(f"❌ Connection error: {e}")
                raise
            
            # Log detailed response info
            logger.info(f"📊 Vector API Response Details:")
            logger.info(f"  Status Code: {response.status_code}")
            logger.info(f"  Reason: {response.reason}")
            logger.info(f"  Headers: {dict(response.headers)}")
            logger.info(f"  Content Length: {len(response.content)}")
            logger.info(f"  Content Type: {response.headers.get('content-type', 'unknown')}")
            
            if response.content:
                content_preview = response.text[:300] if response.text else str(response.content[:300])
                logger.info(f"  Content Preview: {content_preview}...")
            else:
                logger.warning("  ⚠️  Empty response content!")
            
            if response.status_code == 200:
                logger.info("✅ Vector API call successful")
                return response
            else:
                logger.error(f"❌ Vector API returned {response.status_code}")
                logger.error(f"Full response body: {response.text}")
                
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Vector API failed after {MAX_RETRIES} attempts. Status: {response.status_code}, Body: {response.text[:200]}"
                    )
                    
        except requests.exceptions.Timeout as e:
            logger.warning(f"⏰ Timeout on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=504,
                    detail=f"Vector API timeout after {MAX_RETRIES} attempts. The service may be overloaded or sleeping."
                )
                
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            logger.warning(f"🔌 Connection error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=503,
                    detail=f"Vector API connection failed after {MAX_RETRIES} attempts: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"💥 Unexpected error on attempt {attempt + 1}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Vector API failed after {MAX_RETRIES} attempts: {type(e).__name__}: {str(e)}"
                )
        
        # Wait before retry with exponential backoff
        if attempt < MAX_RETRIES - 1:
            wait_time = min((2 ** attempt) + 1, 30)  # Cap at 30 seconds
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    # Should never reach here, but just in case
    raise HTTPException(status_code=500, detail="Vector API failed after all retries")

def convert_vision_coordinates_to_pdf(vision_data: Dict, pdf_page_size: Dict) -> Dict:
    """
    Convert vision coordinates from image pixels to PDF coordinates
    
    Args:
        vision_data: Vision output with image coordinates
        pdf_page_size: {"width": 3370.0, "height": 2384.0} from Vector API
    
    Returns:
        Vision data with converted coordinates
    """
    if not vision_data or not pdf_page_size:
        logger.warning("Missing vision data or PDF page size for coordinate conversion")
        return vision_data
    
    try:
        # Get image metadata from vision output
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
        
        logger.info(f"  Scale factors: X={scale_x:.6f}, Y={scale_y:.6f}")
        
        # Convert regions
        converted_regions = []
        for region in vision_data.get("regions", []):
            coord_block = region.get("coordinate_block", [])
            if len(coord_block) >= 4:
                # Original coordinates in image pixels: [x1, y1, x2, y2]
                x1_img, y1_img, x2_img, y2_img = coord_block
                
                # Convert to PDF points
                x1_pdf = x1_img * scale_x
                y1_pdf = y1_img * scale_y
                x2_pdf = x2_img * scale_x
                y2_pdf = y2_img * scale_y
                
                # Create converted region
                converted_region = region.copy()
                converted_region["coordinate_block_pdf"] = [
                    round(x1_pdf, 2), 
                    round(y1_pdf, 2), 
                    round(x2_pdf, 2), 
                    round(y2_pdf, 2)
                ]
                converted_region["coordinate_block_original"] = coord_block
                converted_regions.append(converted_region)
                
                logger.info(f"  Region '{region.get('label', 'unnamed')}':")
                logger.info(f"    Image: [{x1_img}, {y1_img}, {x2_img}, {y2_img}]")
                logger.info(f"    PDF:   [{x1_pdf:.1f}, {y1_pdf:.1f}, {x2_pdf:.1f}, {y2_pdf:.1f}]")
        
        # Create converted vision data
        converted_vision = vision_data.copy()
        converted_vision["regions"] = converted_regions
        converted_vision["conversion_info"] = {
            "scale_x": round(scale_x, 6),
            "scale_y": round(scale_y, 6),
            "original_image_size": {
                "width_px": image_width_px,
                "height_px": image_height_px
            },
            "pdf_page_size": {
                "width_pts": pdf_width_pts,
                "height_pts": pdf_height_pts
            },
            "conversion_applied": True
        }
        
        logger.info(f"Successfully converted {len(converted_regions)} regions")
        return converted_vision
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return vision_data

@app.post("/process/")
async def process_pdf(
    file: UploadFile = File(...),
    vision_output: str = Form(...)
):
    """
    Process PDF workflow:
    1. Save uploaded PDF
    2. Parse vision_output
    3. Call Vector Drawing API  
    4. Extract PDF page dimensions
    5. Convert vision coordinates to PDF coordinates
    6. Send to Pre-Filter API
    """
    try:
        logger.info(f"=== Starting PDF Processing ===")
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Save uploaded PDF
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"File content read: {len(file_content)} bytes ({file_size_mb:.2f} MB)")
        
        # Save PDF to temporary location
        temp_pdf_path = f"/tmp/uploaded_pdf_{uuid.uuid4()}.pdf"
        with open(temp_pdf_path, 'wb') as f:
            f.write(file_content)
        logger.info(f"PDF saved to: {temp_pdf_path}")
        
        # Step 2: Parse vision_output
        vision_data = None
        if vision_output:
            try:
                vision_data = json.loads(vision_output)
                regions_count = len(vision_data.get('regions', []))
                logger.info(f"Vision output parsed successfully - found {regions_count} regions")
                
                # Log vision regions for debugging
                for i, region in enumerate(vision_data.get('regions', [])):
                    coord_block = region.get('coordinate_block', [])
                    label = region.get('label', 'unnamed')
                    logger.info(f"  Region {i+1}: '{label}' -> {coord_block}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for vision_output: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="vision_output is required")

        # Step 3: Call Vector Drawing API with robust retry
        logger.info("=== Calling Vector Drawing API ===")
        
        params = {
            'minify': 'true',  # As requested
            'remove_non_essential': 'false',
            'precision': '2'
        }
        
        try:
            vector_response = await call_vector_api_with_retry(file_content, file.filename, params)
        except HTTPException as e:
            logger.error(f"Vector API failed after all retries: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error calling Vector API: {e}")
            raise HTTPException(status_code=500, detail=f"Vector API error: {str(e)}")
        
        # Parse vector response
        raw_response = vector_response.text
        logger.info(f"Vector API response length: {len(raw_response)} bytes")
        logger.info(f"Response preview: {raw_response[:100]}...")
        
        try:
            vector_data = json.loads(raw_response)
            logger.info(f"JSON parsed successfully, type: {type(vector_data)}")
            
            # Handle double-encoded JSON
            if isinstance(vector_data, str):
                logger.warning("Response is double-encoded, parsing again")
                vector_data = json.loads(vector_data)
                logger.info(f"Second parse successful, type: {type(vector_data)}")
            
            # Validate structure
            if not isinstance(vector_data, dict):
                raise ValueError("Response is not a dictionary")
            
            if 'pages' not in vector_data or not vector_data['pages']:
                raise ValueError("No pages found in vector data")
                
            logger.info(f"Vector data keys: {list(vector_data.keys())}")
            
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            # Save problematic response for debugging
            debug_path = f"/tmp/vector_debug_{uuid.uuid4()}.json"
            with open(debug_path, 'w') as f:
                f.write(raw_response)
            logger.info(f"Saved debug response to: {debug_path}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector API response: {str(e)}")

        # Step 4: Extract PDF page dimensions
        logger.info("=== Extracting PDF Page Dimensions ===")
        
        first_page = vector_data['pages'][0]
        pdf_page_size = first_page.get('page_size', {})
        
        if not pdf_page_size:
            logger.error("No page_size found in vector data")
            raise HTTPException(status_code=500, detail="No page dimensions found in Vector API response")
        
        logger.info(f"PDF page size: {pdf_page_size}")
        
        # Count extracted elements
        page_texts = len(first_page.get('texts', []))
        page_drawings = first_page.get('drawings', {})
        total_vectors = sum(len(page_drawings.get(key, [])) for key in ['lines', 'rectangles', 'curves', 'polygons'])
        
        logger.info(f"Vector API extracted: {page_texts} texts, {total_vectors} vector elements")

        # Step 5: Convert vision coordinates to PDF coordinates
        logger.info("=== Converting Vision Coordinates ===")
        
        converted_vision = convert_vision_coordinates_to_pdf(vision_data, pdf_page_size)
        
        # Step 6: Prepare data for Pre-Filter API
        logger.info("=== Preparing Data for Pre-Filter API ===")
        
        combined_data = {
            "vision_output": converted_vision,
            "vector_output": vector_data,  # Full raw vector data
            "coordinate_bounds": vector_data.get('summary', {}).get('coordinate_bounds', {}),
            "pdf_dimensions": vector_data.get('metadata', {}).get('pdf_dimensions', {}),
            "metadata": {
                "filename": file.filename,
                "file_size_mb": file_size_mb,
                "total_pages": vector_data.get('summary', {}).get('total_pages', 1),
                "total_texts": vector_data.get('summary', {}).get('total_texts', page_texts),
                "total_vectors": total_vectors,
                "dimensions_found": vector_data.get('summary', {}).get('dimensions_found', 0),
                "coordinate_conversion": "applied",
                "pdf_page_size": pdf_page_size
            }
        }
        
        logger.info("Combined data summary:")
        logger.info(f"  - Vision regions: {len(converted_vision.get('regions', []))} (converted)")
        logger.info(f"  - Vector pages: {len(vector_data.get('pages', []))}")
        logger.info(f"  - PDF page size: {pdf_page_size}")
        
        # Step 7: Call Pre-Filter API
        logger.info("=== Calling Pre-Filter API ===")
        
        try:
            headers = {'Content-Type': 'application/json'}
            
            filter_response = requests.post(
                PRE_FILTER_API_URL,
                json=combined_data,
                headers=headers,
                timeout=300
            )
            
            logger.info(f"Pre-Filter API response status: {filter_response.status_code}")
            
            if filter_response.status_code != 200:
                logger.warning(f"Pre-Filter API error: {filter_response.status_code}")
                logger.warning(f"Error details: {filter_response.text}")
                
                # Return partial success with all data
                return {
                    "status": "partial_success",
                    "message": "Vector extraction and coordinate conversion successful, but pre-filtering failed",
                    "vector_data": vector_data,
                    "vision_data": converted_vision,
                    "scale_data": None,
                    "filter_error": filter_response.text,
                    "processing_stats": {
                        "coordinate_conversion": "applied",
                        "pdf_page_size": pdf_page_size,
                        "regions_converted": len(converted_vision.get('regions', []))
                    },
                    "timestamp": "2025-07-22"
                }
            
            # Parse successful Pre-Filter response
            try:
                filtered_data = filter_response.json()
                logger.info("Pre-Filter API response parsed successfully")
            except Exception as e:
                logger.error(f"Error parsing Pre-Filter response: {e}")
                return {
                    "status": "partial_success",
                    "message": "Processing successful but filter response parsing failed",
                    "vector_data": vector_data,
                    "vision_data": converted_vision,
                    "scale_data": None,
                    "filter_error": f"Parse error: {str(e)}",
                    "timestamp": "2025-07-22"
                }
            
            # Step 8: Return successful results
            logger.info("=== Processing Completed Successfully ===")
            
            return {
                "status": "success",
                "message": "PDF processed successfully through Vector Drawing API and Pre-Filter API",
                "vector_data": vector_data,  # Full raw vector data (for compatibility)
                "scale_data": filtered_data,  # Filtered results (for compatibility)
                "data": filtered_data,  # Main filtered results
                "vision_data": converted_vision,  # Vision data with converted coordinates
                "processing_stats": {
                    "original_texts": page_texts,
                    "total_vectors": total_vectors,
                    "regions_processed": len(converted_vision.get('regions', [])),
                    "coordinate_conversion": "applied",
                    "pdf_page_size": pdf_page_size,
                    "conversion_info": converted_vision.get('conversion_info', {})
                },
                "timestamp": "2025-07-22"
            }
            
        except Exception as e:
            logger.error(f"Error calling Pre-Filter API: {e}")
            return {
                "status": "partial_success",
                "message": "Vector extraction and coordinate conversion successful, but pre-filtering failed",
                "vector_data": vector_data,
                "vision_data": converted_vision,
                "scale_data": None,
                "error": f"Pre-Filter API error: {str(e)}",
                "timestamp": "2025-07-22"
            }
        
        finally:
            # Clean up temporary PDF file
            try:
                os.unlink(temp_pdf_path)
                logger.info(f"Cleaned up temporary file: {temp_pdf_path}")
            except:
                pass
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.2.1",
        "features": [
            "PDF upload and temporary storage",
            "Vision output parsing with region extraction",
            "Vector Drawing API integration with robust retry logic",
            "Precise coordinate conversion (image pixels -> PDF points)",
            "Pre-Filter API integration with converted coordinates",
            "Full raw vector data passthrough",
            "IncompleteRead error handling"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Master API",
        "version": "1.2.1",
        "description": "Processes PDF with precise coordinate conversion and robust error handling",
        "workflow": [
            "1. Receive PDF file and vision_output via multipart form",
            "2. Save PDF and parse vision regions with image coordinates", 
            "3. Call Vector Drawing API to extract all vector data",
            "4. Extract PDF page dimensions (width, height in points)",
            "5. Convert vision coordinates from image pixels to PDF points",
            "6. Send converted vision_output + raw vector_output to Pre-Filter API",
            "7. Return combined results with conversion statistics"
        ],
        "coordinate_conversion": {
            "input": "Vision regions in image pixels",
            "output": "Vision regions in PDF points", 
            "method": "Scale factors based on image size vs PDF page size",
            "example": {
                "image_size": {"width": 9969, "height": 7052},
                "pdf_size": {"width": 3370.0, "height": 2384.0},
                "scale_factors": {"x": 0.338, "y": 0.338}
            }
        },
        "apis_used": {
            "vector_drawing": VECTOR_API_URL,
            "pre_filter": PRE_FILTER_API_URL
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Master API v1.2.1 on port {PORT}")
    logger.info("Features: Robust Vector API calls, coordinate conversion, IncompleteRead error handling")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
