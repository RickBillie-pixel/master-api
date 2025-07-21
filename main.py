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

VECTOR_API_URL = "https://vector-drawing.onrender.com/extract/"  # Correcte URL
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-scale"

@app.post("/process/")
async def process_pdf(file: UploadFile, vision_output: str = Form(...)):
    """Process PDF: Extract vectors via Vector Drawing API, combine data, and filter via Pre-Filter API"""
    filtered_data = None
    vector_data = None
    
    try:
        logger.info(f"Received file: {file.filename} and vision_output")

        # Stap 2: Ontvang en parse vision_output
        try:
            vision_output_dict = json.loads(vision_output)
            logger.info(f"Parsed vision_output: {vision_output_dict}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in vision_output: {str(e)}")

        # Stap 3: Aanroepen VectorDrawing-API
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'false',
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
        
        # Parse vector response
        raw_response = vector_response.text
        try:
            vector_data = json.loads(raw_response)
            if isinstance(vector_data, str):
                vector_data = json.loads(vector_data)
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector Drawing API response: {str(e)}")
        
        # Valideer structuur
        if not isinstance(vector_data, dict) or 'pages' not in vector_data or 'metadata' not in vector_data:
            raise HTTPException(status_code=400, detail="Invalid Vector Drawing API response structure")

        # Stap 4: Combineer data
        combined_data = {
            "vision_output": vision_output_dict,
            "vector_output": vector_data,
            "page_size": vector_data.get('page_size', {'width': 0, 'height': 0}),
            "bounding_box": vector_data.get('bbox', [0, 0, 0, 0]),
            "page_number": vector_data.get('page_number', 1)
        }
        logger.info(f"Combined data: {combined_data}")

        # Stap 5: Stuur naar PreFilter API
        logger.info("Calling Pre-Filter API")
        headers = {'Content-Type': 'application/json'}
        filter_response = requests.post(
            PRE_FILTER_API_URL,
            json=combined_data,
            headers=headers,
            timeout=300
        )
        
        if filter_response.status_code != 200:
            logger.error(f"Pre-Filter API error: {filter_response.status_code} - {filter_response.text}")
            return {
                "status": "partial_success",
                "message": "Vector extraction successful, but scale filtering failed",
                "data": combined_data,
                "filter_error": filter_response.text
            }
        
        filtered_data = filter_response.json()
        logger.info("Pre-Filter API response parsed successfully")

        return {
            "status": "success",
            "message": "PDF processed successfully through both APIs",
            "data": filtered_data
        }

    except HTTPException as e:
        raise e
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
