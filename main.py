import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, HTTPException
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
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API",
    version="1.0.2"
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
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

@app.post("/process/")
async def process_pdf(file: UploadFile):
    """Process PDF: Extract vectors via Vector Drawing API, then filter via Pre-Filter API"""
    filtered_data = None
    vector_data = None
    
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Call Vector Drawing API with specified parameters
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'true',  # Use non-minified output for easier processing
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
        
        # Get the raw response text
        raw_response = vector_response.text
        logger.info(f"Vector Drawing API response length: {len(raw_response)} bytes")
        
        # Parse the JSON response
        try:
            vector_data = json.loads(raw_response)
            
            # Handle double-encoded JSON if necessary
            if isinstance(vector_data, str):
                logger.warning("Response is double-encoded, parsing again")
                vector_data = json.loads(vector_data)
            
            logger.info(f"Parsed Vector Drawing API response successfully")
            
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector Drawing API response: {str(e)}")
        
        # Validate structure
        if not isinstance(vector_data, dict):
            raise HTTPException(status_code=400, detail="Vector Drawing API response is not a dictionary")
            
        if 'pages' not in vector_data or 'metadata' not in vector_data:
            raise HTTPException(status_code=400, detail="Vector Drawing API response missing required fields")
        
        # Step 2: Send the complete Vector Drawing API output to Pre-Filter API
        # Send JSON data directly instead of file upload
        try:
            logger.info("Calling Pre-Filter API with JSON data")
            headers = {'Content-Type': 'application/json'}
            filter_response = requests.post(
                PRE_FILTER_API_URL, 
                json=vector_data,  # Send JSON directly
                headers=headers,
                timeout=300
            )
            
            logger.info(f"Pre-Filter API response status: {filter_response.status_code}")
            
            if filter_response.status_code != 200:
                logger.error(f"Pre-Filter API error: {filter_response.status_code} - {filter_response.text}")
                # Return original vector data if filtering fails
                logger.warning("Returning original vector data due to filter API failure")
                return {
                    "status": "partial_success",
                    "message": "Vector extraction successful, but scale filtering failed",
                    "data": vector_data,
                    "filter_error": filter_response.text
                }
            
            # Parse Pre-Filter API response
            try:
                filtered_data = filter_response.json()
                logger.info("Pre-Filter API response parsed successfully")
                
                # Log filtering statistics
                if 'processing_stats' in filtered_data:
                    stats = filtered_data['processing_stats']
                    logger.info(f"Processing stats: {stats}")
                
                # Return the filtered data with success status
                return {
                    "status": "success",
                    "message": "PDF processed successfully through both APIs",
                    "data": filtered_data
                }
                
            except Exception as e:
                logger.error(f"Error parsing Pre-Filter API response: {e}")
                # Return original vector data if filter parsing fails
                return {
                    "status": "partial_success",
                    "message": "Vector extraction successful, but scale filter response parsing failed",
                    "data": vector_data,
                    "filter_error": str(e)
                }
            
        except requests.exceptions.Timeout:
            logger.error("Pre-Filter API timeout")
            return {
                "status": "partial_success",
                "message": "Vector extraction successful, but scale filtering timed out",
                "data": vector_data,
                "filter_error": "Timeout"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Pre-Filter API request error: {e}")
            return {
                "status": "partial_success",
                "message": "Vector extraction successful, but scale filtering failed",
                "data": vector_data,
                "filter_error": str(e)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0.2"}

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
