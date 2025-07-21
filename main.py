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

VECTOR_API_URL = "https://vector-drawing.onrender.com/extract/"
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

@app.post("/process/")
async def process_pdf(file: UploadFile, vision_output: str = Form(...)):
    """Process PDF: Extract vectors via Vector Drawing API, combine with vision output, then filter via Pre-Filter API"""
    filtered_data = None
    vector_data = None
    
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Parse vision_output
        try:
            vision_output_dict = json.loads(vision_output)
            logger.info("Vision output parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for vision_output: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")
        
        # Step 2: Call Vector Drawing API
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'true',  # Minify aan, zoals je vroeg
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
        if not isinstance(vector_data, dict):
            raise HTTPException(status_code=400, detail="Vector Drawing API response is not a dictionary")
            
        if 'pages' not in vector_data or 'metadata' not in vector_data:
            raise HTTPException(status_code=400, detail="Vector Drawing API response missing required fields")
        
        # Step 3: Combine data
        combined_data = {
            "vision_output": vision_output_dict,
            "vector_output": vector_data,
            "page_size": vector_data.get('metadata', {}).get('pdf_dimensions', {}).get('max_width_points', 0),  # Voorbeeld, pas aan
            "bounding_box": vector_data.get('summary', {}).get('coordinate_bounds', [0, 0, 0, 0])
        }
        logger.info("Data combined successfully")
        
        # Step 4: Send to Pre-Filter API
        logger.info("Calling Pre-Filter API with JSON data")
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
            # Return partial success
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
    return {"status": "healthy", "version": "1.0.3"}

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
