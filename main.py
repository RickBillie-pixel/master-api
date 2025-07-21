import os
import tempfile
import uuid
import json
import logging
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
from typing import Dict, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API",
    version="1.0.3"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API URLs - Updated with correct endpoints
VECTOR_API_URL = "https://vector-drawing.onrender.com/extract/"
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

# Add timeout and retry logic
REQUEST_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3

async def call_api_with_retry(url: str, method: str = "POST", **kwargs):
    """Call API with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempting API call to {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            
            if method == "POST":
                response = requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
            else:
                response = requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            
            return response
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=504, detail=f"API timeout after {MAX_RETRIES} attempts")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt + 1} for {url}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=503, detail=f"Service unavailable after {MAX_RETRIES} attempts")
            await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1} for {url}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
            await asyncio.sleep(1)

@app.post("/process/")
async def process_pdf(
    file: UploadFile = File(...),
    vision_output: str = Form(...)
):
    """Process PDF: Extract vectors via Vector Drawing API, combine data, and filter via Pre-Filter API"""
    
    try:
        logger.info(f"=== Starting PDF Processing ===")
        logger.info(f"Received file: {file.filename}")
        logger.info(f"File size: {file.size if hasattr(file, 'size') else 'unknown'} bytes")
        logger.info(f"Vision output length: {len(vision_output)} chars")

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
            
            logger.info(f"File content read successfully: {len(file_content)} bytes")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

        # Step 4: Call Vector Drawing API
        logger.info("=== Calling Vector Drawing API ===")
        
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'false',
            'remove_non_essential': 'false',
            'precision': '2'
        }
        
        try:
            vector_response = await call_api_with_retry(
                VECTOR_API_URL,
                method="POST",
                files=files,
                params=params
            )
            
            logger.info(f"Vector API response status: {vector_response.status_code}")
            
            if vector_response.status_code != 200:
                logger.error(f"Vector API error: {vector_response.status_code} - {vector_response.text}")
                raise HTTPException(
                    status_code=vector_response.status_code,
                    detail=f"Vector Drawing API error: {vector_response.text}"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vector API call failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to call Vector Drawing API: {str(e)}")

        # Step 5: Parse vector response
        try:
            raw_response = vector_response.text
            logger.info(f"Vector API response length: {len(raw_response)} chars")
            
            # Handle both string and dict responses
            if raw_response.startswith('"') and raw_response.endswith('"'):
                # Response is a JSON string that needs double parsing
                vector_data = json.loads(json.loads(raw_response))
            else:
                vector_data = json.loads(raw_response)
            
            logger.info("Vector API response parsed successfully")
            
            # Validate response structure
            if not isinstance(vector_data, dict):
                raise ValueError("Response is not a dictionary")
            
            if 'pages' not in vector_data:
                logger.warning("No 'pages' key in vector response")
            
            if 'metadata' not in vector_data:
                logger.warning("No 'metadata' key in vector response")
                
        except Exception as e:
            logger.error(f"Vector API response parsing error: {str(e)}")
            logger.error(f"Raw response preview: {raw_response[:500]}...")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse Vector Drawing API response: {str(e)}"
            )

        # Step 6: Extract coordinate bounds and page info
        try:
            # Get coordinate bounds from summary
            coordinate_bounds = vector_data.get('summary', {}).get('coordinate_bounds', {})
            
            # Get page dimensions
            page_dimensions = {}
            if 'pages' in vector_data and len(vector_data['pages']) > 0:
                first_page = vector_data['pages'][0]
                page_dimensions = first_page.get('page_dimensions', {})
                if not page_dimensions and 'page_size' in first_page:
                    page_dimensions = first_page['page_size']
            
            logger.info(f"Coordinate bounds: {coordinate_bounds}")
            logger.info(f"Page dimensions: {page_dimensions}")
            
        except Exception as e:
            logger.warning(f"Error extracting coordinate info: {str(e)}")
            coordinate_bounds = {}
            page_dimensions = {}

        # Step 7: Combine data for Pre-Filter API
        combined_data = {
            "vision_output": vision_output_dict,
            "vector_output": vector_data,
            "coordinate_bounds": coordinate_bounds,
            "page_dimensions": page_dimensions,
            "metadata": {
                "filename": file.filename,
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
                        "vector_summary": {
                            "total_texts": vector_data.get('summary', {}).get('total_texts', 0),
                            "coordinate_bounds": coordinate_bounds
                        }
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
                    "vector_summary": {
                        "total_texts": vector_data.get('summary', {}).get('total_texts', 0),
                        "coordinate_bounds": coordinate_bounds
                    }
                }
            )

        # Step 9: Return successful result
        logger.info("=== PDF Processing Completed Successfully ===")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "PDF processed successfully through both APIs",
                "data": filtered_data,
                "metadata": {
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
                "pre_filter_api": "unknown"
            }
        }
        
        # Quick health check for Vector API
        try:
            vector_health_response = requests.get(
                "https://vector-drawing.onrender.com/health/",
                timeout=10
            )
            health_status["services"]["vector_api"] = "healthy" if vector_health_response.status_code == 200 else "unhealthy"
        except:
            health_status["services"]["vector_api"] = "unreachable"
        
        # Quick health check for Pre-Filter API  
        try:
            filter_health_response = requests.get(
                "https://pre-filter-scale-api.onrender.com/health/",
                timeout=10
            )
            health_status["services"]["pre_filter_api"] = "healthy" if filter_health_response.status_code == 200 else "unhealthy"
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
        "version": "1.0.3",
        "description": "Processes PDFs through Vector Drawing API and Pre-Filter API",
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
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Master API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
