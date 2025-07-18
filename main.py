import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Scale API",
    version="1.0.0"
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
SCALE_API_URL = "https://scale-api-69gl.onrender.com/extract-scale/"

@app.post("/process/")
async def process_pdf(file: UploadFile):
    """Process PDF: Extract vectors via Vector Drawing API, then calculate scale via Scale API"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Call Vector Drawing API with specified parameters
        files = {'file': (file.filename, await file.read(), 'application/pdf')}
        params = {
            'minify': 'true',
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
        
        # Parse the minified JSON response and log raw text for debugging
        raw_response = vector_response.text
        logger.info("Raw Vector Drawing API response: %s", raw_response[:500])  # Log first 500 chars
        try:
            vector_data = vector_response.json()
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse Vector Drawing API response as JSON: {e}"
            )
        
        logger.info("Vector Drawing API response received: %s", vector_data)
        
        # Extract relevant data for Scale API (first page assumed)
        if not vector_data.get('pages'):
            raise HTTPException(status_code=400, detail="No pages in vector data")
        
        page = vector_data['pages'][0]
        drawings = page.get('drawings', {})
        texts = page.get('texts', [])
        
        # Ensure vectors have length if missing
        vector_data_for_scale = {
            "vector_data": [
                {
                    "type": v["type"],
                    "p1": v["p1"],
                    "p2": v["p2"],
                    "length": v.get("length", None)  # Will be inferred by Scale API if None
                } for v in drawings.get('lines', []) + drawings.get('curves', [])
            ],
            "texts": [
                {"text": t["text"], "position": t["position"]} for t in texts
            ]
        }
        
        # Step 2: Call Scale API
        logger.info("Calling Scale API with data: %s", vector_data_for_scale)
        scale_response = requests.post(SCALE_API_URL, json=vector_data_for_scale, timeout=300)
        
        if scale_response.status_code != 200:
            raise HTTPException(
                status_code=scale_response.status_code,
                detail=f"Scale API error: {scale_response.text}"
            )
        
        scale_data = scale_response.json()
        logger.info("Scale API response received: %s", scale_data)
        
        # Combine and return results
        result = {
            "vector_data": vector_data,
            "scale_data": scale_data,
            "timestamp": "2025-07-18 17:50 CEST"
        }
        
        return result
        
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0", "timestamp": "2025-07-18 17:50 CEST"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
