import os
import tempfile
import uuid
import json
import logging
import time
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
from requests.adapters import HTTPAdapter
from typing import Dict, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API with vision output",
    version="1.0.5"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED URLs - using correct vector API URL
VECTOR_API_URL = "https://vector-drawing.onrender.com/extract/"  # Fixed spelling
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

# Enhanced timeout and retry settings
REQUEST_TIMEOUT = 600  # 10 minutes for large PDFs
CONNECT_TIMEOUT = 30   # 30 seconds for connection
MAX_RETRIES = 3

def create_robust_session():
    """Create a requests session with basic retry strategy"""
    session = requests.Session()
    
    # Simple adapter without complex retry strategy to avoid version conflicts
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        pool_block=False
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

async def call_vector_api_robust(file_content: bytes, filename: str, params: dict):
    """Robust API call specifically for Vector Drawing API"""
    
    for attempt in range(MAX_RETRIES):
        session = None
        try:
            logger.info(f"Vector API attempt {attempt + 1}/{MAX_RETRIES}")
            logger.info(f"File size: {len(file_content)} bytes, Filename: {filename}")
            
            # Create robust session
            session = create_robust_session()
            
            # Prepare files and data
            files = {'file': (filename, file_content, 'application/pdf')}
            
            # Set comprehensive headers
            headers = {
                'User-Agent': 'Master-API/1.0.5',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            logger.info(f"Making POST request to {VECTOR_API_URL}")
            logger.info(f"Params: {params}")
            
            # Make request with extended timeouts
            response = session.post(
                VECTOR_API_URL,
                files=files,
                params=params,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
                stream=False,  # Don't stream to avoid incomplete reads
                verify=True
            )
            
            logger.info(f"Vector API response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                logger.info(f"Success! Response size: {len(response.content)} bytes")
                return response
            else:
                logger.error(f"Non-200 status: {response.status_code}")
                logger.error(f"Response text: {response.text[:500]}...")
                
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Vector API failed with status {response.status_code}: {response.text[:200]}"
                    )
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=504, 
                    detail=f"Vector API timeout after {MAX_RETRIES} attempts. Large PDF may need more time."
                )
                
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Vector API connection failed after {MAX_RETRIES} attempts: {str(e)}"
                )
                
        except requests.exceptions.ChunkedEncodingError as e:
            logger.warning(f"Chunked encoding error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=502, 
                    detail=f"Vector API response incomplete after {MAX_RETRIES} attempts"
                )
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Vector API failed: {str(e)}"
                )
        
        finally:
            if session:
                session.close()
        
        # Wait before retry
        if attempt < MAX_RETRIES - 1:
            wait_time = (2 ** attempt) + 1  # Exponential backoff + 1
            logger.info(f"Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)

@app.post("/process/")
async def process_pdf(
    file: UploadFile = File(...),
    vision_output: str = Form(...)  # NEW: Added vision_output parameter
):
    """Process PDF: Extract vectors via Vector Drawing API, combine with vision output, and filter via Pre-Filter API"""
    
    start_time = time.time()
    
    try:
        logger.info(f"=== Starting PDF Processing ===")
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Step 2: Parse and validate vision_output
        try:
            vision_output_dict = json.loads(vision_output)
            logger.info(f"Successfully parsed vision_output with {len(vision_output_dict)} keys")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in vision_output: {str(e)}")

        # Step 3: Read file content
        try:
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file received")
            
            file_size_mb = len(file_content) / (1024 * 1024)
            logger.info(f"File content read successfully: {len(file_content)} bytes ({file_size_mb:.2f} MB)")
            
            # Warn about large files
            if file_size_mb > 5:
                logger.warning(f"Large PDF detected ({file_size_mb:.2f} MB) - processing may take longer")
                
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

        # Step 4: Call Vector Drawing API with robust handling
        logger.info("=== Calling Vector Drawing API ===")
        
        params = {
            'minify': 'false',  # Use non-minified output like old version
            'remove_non_essential': 'false',
            'precision': '2'
        }
        
        try:
            vector_response = await call_vector_api_robust(file_content, file.filename, params)
            
            if not vector_response:
                raise HTTPException(status_code=500, detail="Vector API returned no response")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vector API call failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to call Vector Drawing API: {str(e)}")

        # Step 5: Parse vector response (using same logic as old version)
        try:
            raw_response = vector_response.text
            logger.info(f"Raw Vector Drawing API response (first 100 chars): {raw_response[:100]}")
            logger.info(f"Response length: {len(raw_response)} bytes")
            
            # Parse JSON (same as old version)
            vector_data = json.loads(raw_response)
            logger.info(f"JSON parsed successfully, type={type(vector_data)}")
            logger.info(f"Top-level keys: {list(vector_data.keys()) if isinstance(vector_data, dict) else 'Not a dict'}")
            
            # Double-check if we need to parse again (in case of double-encoding)
            if isinstance(vector_data, str):
                logger.warning("Parsed result is still a string, attempting to parse again")
                vector_data = json.loads(vector_data)
                logger.info(f"Second parse: type={type(vector_data)}")
                logger.info(f"Second parse keys: {list(vector_data.keys()) if isinstance(vector_data, dict) else 'Not a dict'}")
            
            # Validate response structure
            if not isinstance(vector_data, dict):
                raise HTTPException(status_code=400, detail="Parsed result is not a dictionary")
                
            if 'pages' not in vector_data or not vector_data['pages']:
                raise HTTPException(status_code=400, detail="No pages found in vector data")
                
        except Exception as e:
            logger.error(f"Vector API response parsing error: {str(e)}")
            # Save problematic response for debugging
            debug_path = f"/tmp/vector_response_{uuid.uuid4()}.json"
            with open(debug_path, 'w') as f:
                f.write(raw_response)
            logger.info(f"Saved problematic response to {debug_path}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse Vector Drawing API response: {str(e)}"
            )

        # Step 6: Extract coordinate bounds and page info
        try:
            coordinate_bounds = vector_data.get('summary', {}).get('coordinate_bounds', {})
            pdf_dimensions = vector_data.get('metadata', {}).get('pdf_dimensions', {})
            
            logger.info(f"Coordinate bounds: {coordinate_bounds}")
            logger.info(f"PDF dimensions: {pdf_dimensions}")
            
        except Exception as e:
            logger.warning(f"Error extracting coordinate info: {str(e)}")
            coordinate_bounds = {}
            pdf_dimensions = {}

        # Step 7: Combine data for Pre-Filter API
        combined_data = {
            "vision_output": vision_output_dict,  # NEW: Include vision output
            "vector_output": vector_data,
            "coordinate_bounds": coordinate_bounds,
            "pdf_dimensions": pdf_dimensions,
            "metadata": {
                "filename": file.filename,
                "file_size_mb": file_size_mb,
                "processing_time_vector_ms": int((time.time() - start_time) * 1000),
                "total_pages": vector_data.get('summary', {}).get('total_pages', 1),
                "total_texts": vector_data.get('summary', {}).get('total_texts', 0),
                "dimensions_found": vector_data.get('summary', {}).get('dimensions_found', 0)
            }
        }
        
        logger.info("=== Data combined successfully ===")

        # Step 8: Call Pre-Filter API
        logger.info("=== Calling Pre-Filter API ===")
        
        try:
            session = create_robust_session()
            headers = {'Content-Type': 'application/json'}
            
            filter_response = session.post(
                PRE_FILTER_API_URL,
                json=combined_data,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT)
            )
            
            session.close()
            
            logger.info(f"Pre-Filter API response status: {filter_response.status_code}")
            
            if filter_response.status_code != 200:
                logger.error(f"Pre-Filter API error: {filter_response.status_code} - {filter_response.text}")
                # Return partial success with vector data
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "partial_success",
                        "message": "Vector extraction successful, but scale filtering failed",
                        "vector_data": vector_data,
                        "scale_data": None,
                        "filter_error": filter_response.text,
                        "processing_time_ms": int((time.time() - start_time) * 1000)
                    }
                )
            
            filtered_data = filter_response.json()
            logger.info("Pre-Filter API response parsed successfully")

        except Exception as e:
            logger.error(f"Pre-Filter API call failed: {str(e)}")
            # Return partial success with vector data (like old version)
            return JSONResponse(
                status_code=200,
                content={
                    "status": "partial_success",
                    "message": "Vector extraction successful, but scale filtering failed",
                    "vector_data": vector_data,
                    "scale_data": None,
                    "error": str(e),
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            )

        # Step 9: Return successful result (compatible with old format)
        total_time = int((time.time() - start_time) * 1000)
        logger.info(f"=== PDF Processing Completed Successfully in {total_time}ms ===")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "PDF processed successfully through both APIs",
                "vector_data": vector_data,  # Keep old format compatibility
                "scale_data": filtered_data,  # Keep old format compatibility
                "data": filtered_data,  # New format
                "timestamp": "2025-07-21",
                "processing_time_ms": total_time,
                "file_size_mb": file_size_mb
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy", 
            "version": "1.0.5",
            "features": [
                "Vision output integration",
                "Robust error handling", 
                "Backward compatibility",
                "Extended timeouts"
            ]
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Master API",
        "version": "1.0.5",
        "description": "Processes PDFs through Vector Drawing API and Pre-Filter API with vision output support",
        "endpoints": {
            "/": "This page",
            "/process/": "POST - Process PDF file with vision output",
            "/health/": "GET - Health check"
        },
        "external_services": {
            "vector_drawing_api": VECTOR_API_URL,
            "pre_filter_api": PRE_FILTER_API_URL
        },
        "request_format": {
            "method": "POST",
            "content_type": "multipart/form-data",
            "fields": {
                "file": "PDF file (binary)",
                "vision_output": "JSON string with vision analysis"
            }
        },
        "improvements": {
            "added_vision_output": "Now accepts vision_output parameter",
            "fixed_vector_api_url": "Corrected vector API URL spelling",
            "robust_error_handling": "Handles IncompleteRead and connection errors",
            "backward_compatibility": "Maintains old response format",
            "extended_timeouts": "10 minute timeout for large PDFs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Enhanced Master API on port {PORT}")
    logger.info("Features: Vision output support, robust error handling, backward compatibility")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

async def call_api_with_retry(url: str, method: str = "POST", **kwargs):
    """Generic API call with retry logic"""
    for attempt in range(MAX_RETRIES):
        session = None
        try:
            logger.info(f"API call to {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            
            session = create_robust_session()
            
            if method == "POST":
                response = session.post(url, timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT), **kwargs)
            else:
                response = session.get(url, timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT), **kwargs)
            
            return response
            
        except Exception as e:
            logger.error(f"API call error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
        
        finally:
            if session:
                session.close()
        
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)

@app.post("/process/")
async def process_pdf(
    file: UploadFile = File(...),
    vision_output: str = Form(...)
):
    """Process PDF: Extract vectors via Vector Drawing API, combine data, and filter via Pre-Filter API"""
    
    start_time = time.time()
    
    try:
        logger.info(f"=== Starting PDF Processing ===")
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Step 2: Parse and validate vision_output
        try:
            vision_output_dict = json.loads(vision_output)
            logger.info(f"Successfully parsed vision_output with {len(vision_output_dict)} keys")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in vision_output: {str(e)}")

        # Step 3: Read file content
        try:
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file received")
            
            file_size_mb = len(file_content) / (1024 * 1024)
            logger.info(f"File content read successfully: {len(file_content)} bytes ({file_size_mb:.2f} MB)")
            
            # Warn about large files
            if file_size_mb > 5:
                logger.warning(f"Large PDF detected ({file_size_mb:.2f} MB) - processing may take longer")
                
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

        # Step 4: Call Vector Drawing API with robust handling
        logger.info("=== Calling Vector Drawing API ===")
        
        params = {
            'minify': 'true',
            'remove_non_essential': 'false',
            'precision': '2'
        }
        
        try:
            vector_response = await call_vector_api_robust(file_content, file.filename, params)
            
            if not vector_response:
                raise HTTPException(status_code=500, detail="Vector API returned no response")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vector API call failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to call Vector Drawing API: {str(e)}")

        # Step 5: Parse vector response
        try:
            raw_response = vector_response.text
            logger.info(f"Vector API response length: {len(raw_response)} chars")
            
            # Handle minified JSON response
            if raw_response.startswith('"') and raw_response.endswith('"'):
                # Response is a JSON string that needs double parsing
                vector_data = json.loads(json.loads(raw_response))
            else:
                vector_data = json.loads(raw_response)
            
            logger.info("Vector API response parsed successfully")
            
            # Validate response structure
            if not isinstance(vector_data, dict):
                raise ValueError("Response is not a dictionary")
                
        except Exception as e:
            logger.error(f"Vector API response parsing error: {str(e)}")
            logger.error(f"Raw response preview: {raw_response[:500]}...")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse Vector Drawing API response: {str(e)}"
            )

        # Step 6: Extract coordinate bounds and page info
        try:
            coordinate_bounds = vector_data.get('summary', {}).get('coordinate_bounds', {})
            pdf_dimensions = vector_data.get('metadata', {}).get('pdf_dimensions', {})
            
            logger.info(f"Coordinate bounds: {coordinate_bounds}")
            logger.info(f"PDF dimensions: {pdf_dimensions}")
            
        except Exception as e:
            logger.warning(f"Error extracting coordinate info: {str(e)}")
            coordinate_bounds = {}
            pdf_dimensions = {}

        # Step 7: Combine data for Pre-Filter API
        combined_data = {
            "vision_output": vision_output_dict,
            "vector_output": vector_data,
            "coordinate_bounds": coordinate_bounds,
            "pdf_dimensions": pdf_dimensions,
            "metadata": {
                "filename": file.filename,
                "file_size_mb": file_size_mb,
                "processing_time_vector_ms": int((time.time() - start_time) * 1000),
                "total_pages": vector_data.get('summary', {}).get('total_pages', 1),
                "total_texts": vector_data.get('summary', {}).get('total_texts', 0),
                "dimensions_found": vector_data.get('summary', {}).get('dimensions_found', 0)
            }
        }
        
        logger.info("=== Data combined successfully ===")

        # Step 8: Call Pre-Filter API
        logger.info("=== Calling Pre-Filter API ===")
        
        try:
            headers = {'Content-Type': 'application/json'}
            filter_response = await call_api_with_retry(
                PRE_FILTER_API_URL,
                method="POST",
                json=combined_data,
                headers=headers
            )
            
            logger.info(f"Pre-Filter API response status: {filter_response.status_code}")
            
            if filter_response.status_code != 200:
                logger.error(f"Pre-Filter API error: {filter_response.status_code} - {filter_response.text}")
                # Return partial success with vector data
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "partial_success",
                        "message": "Vector extraction successful, but scale filtering failed",
                        "data": combined_data,
                        "filter_error": filter_response.text,
                        "processing_time_ms": int((time.time() - start_time) * 1000)
                    }
                )
            
            filtered_data = filter_response.json()
            logger.info("Pre-Filter API response parsed successfully")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Pre-Filter API call failed: {str(e)}")
            # Return partial success with vector data
            return JSONResponse(
                status_code=200,
                content={
                    "status": "partial_success",
                    "message": "Vector extraction successful, but scale filtering failed",
                    "data": combined_data,
                    "filter_error": str(e),
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            )

        # Step 9: Return successful result
        total_time = int((time.time() - start_time) * 1000)
        logger.info(f"=== PDF Processing Completed Successfully in {total_time}ms ===")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "PDF processed successfully through both APIs",
                "data": filtered_data,
                "metadata": {
                    "processing_time_ms": total_time,
                    "file_size_mb": file_size_mb,
                    "processing_summary": {
                        "total_texts": vector_data.get('summary', {}).get('total_texts', 0),
                        "dimensions_found": vector_data.get('summary', {}).get('dimensions_found', 0),
                        "coordinate_bounds": coordinate_bounds
                    }
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    """Enhanced health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "version": "1.0.4",
            "services": {
                "vector_api": "unknown",
                "pre_filter_api": "unknown"
            },
            "capabilities": {
                "max_pdf_size_mb": 10,
                "timeout_minutes": 10,
                "retry_attempts": MAX_RETRIES
            }
        }
        
        # Quick health check for Vector API
        try:
            session = create_robust_session()
            vector_health_response = session.get(
                "https://vector-drawing.onrender.com/health/",
                timeout=10
            )
            health_status["services"]["vector_api"] = "healthy" if vector_health_response.status_code == 200 else "unhealthy"
            session.close()
        except:
            health_status["services"]["vector_api"] = "unreachable"
        
        # Quick health check for Pre-Filter API  
        try:
            session = create_robust_session()
            filter_health_response = session.get(
                "https://pre-filter-scale-api.onrender.com/health/",
                timeout=10
            )
            health_status["services"]["pre_filter_api"] = "healthy" if filter_health_response.status_code == 200 else "unhealthy"
            session.close()
        except:
            health_status["services"]["pre_filter_api"] = "unreachable"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Master API",
        "version": "1.0.4",
        "description": "Processes PDFs through Vector Drawing API and Pre-Filter API with robust error handling",
        "endpoints": {
            "/": "This page",
            "/process/": "POST - Process PDF file with vision output",
            "/health/": "GET - Health check with service status"
        },
        "external_services": {
            "vector_drawing_api": VECTOR_API_URL,
            "pre_filter_api": PRE_FILTER_API_URL
        },
        "request_format": {
            "method": "POST",
            "content_type": "multipart/form-data",
            "fields": {
                "file": "PDF file (binary)",
                "vision_output": "JSON string with vision analysis"
            }
        },
        "improvements": {
            "robust_session": "Uses connection pooling and retry strategies",
            "timeout_handling": "Extended timeouts for large PDFs (10 minutes)",
            "error_recovery": "Handles IncompleteRead and connection errors",
            "partial_success": "Returns vector data even if filtering fails"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Enhanced Master API on port {PORT}")
    logger.info("Features: Robust error handling, extended timeouts, connection pooling")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
