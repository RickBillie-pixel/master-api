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
    version="1.0.1"
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
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Call Vector Drawing API with specified parameters
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'false',  # Use non-minified output for easier processing
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
        # The Pre-Filter API expects the full Vector Drawing API output format
        temp_file_path = f"/tmp/vector_output_{uuid.uuid4()}.json"
        try:
            # Save Vector Drawing API output to temporary file
            with open(temp_file_path, 'w') as f:
                json.dump(vector_data, f)
            
            logger.info(f"Saved Vector Drawing API output to {temp_file_path}")
            
            # Send to Pre-Filter API
            with open(temp_file_path, 'rb') as f:
                filter_files = {'file': (f'vector_output_{file.filename}.json', f, 'application/json')}
                logger.info("Calling Pre-Filter API")
                filter_response = requests.post(PRE_FILTER_API_URL, files=filter_files, timeout=300)
            
            if filter_response.status_code != 200:
                logger.error(f"Pre-Filter API error: {filter_response.status_code} - {filter_response.text}")
                # Return ONLY the filtered data
                return filtered_data
            
            # Parse Pre-Filter API response
            try:
                filtered_data = filter_response.json()
                logger.info("Pre-Filter API response parsed successfully")
                
                # Log filtering statistics
                if 'summary' in filtered_data:
                    summary = filtered_data['summary']
                    logger.info(f"Filtering stats: {summary.get('total_lines', 0)} lines kept out of {summary.get('original_lines', 0)}")
                
            except Exception as e:
                logger.error(f"Error parsing Pre-Filter API response: {e}")
                # Return ONLY the filtered data (even on parse error)
                return vector_data  # Return original if filter parsing fails
            
            # Return ONLY the filtered data
            return filtered_data
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0.1"}

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
