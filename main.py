import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
from typing import Dict

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
    """Process PDF: Extract vectors then calculate scale"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Call Vector Drawing API
        files = {'file': (file.filename, await file.read(), 'application/pdf')}
        params = {
            'minify': 'true',
            'remove_non_essential': 'false',
            'precision': '2'
        }
        
        logger.info("Calling Vector API")
        vector_response = requests.post(VECTOR_API_URL, files=files, params=params)
        
        if vector_response.status_code != 200:
            raise HTTPException(
                status_code=vector_response.status_code,
                detail=f"Vector API error: {vector_response.text}"
            )
        
        vector_data = vector_response.json()
        logger.info("Vector API response received")
        
        # Extract relevant data from vector output
        # Assuming vector_data has 'pages' with 'drawings' and 'texts'
        # We take first page for simplicity; adjust if multi-page
        if not vector_data.get('pages'):
            raise HTTPException(status_code=400, detail="No pages in vector data")
        
        page = vector_data['pages'][0]
        drawings = page.get('drawings', {})
        input_for_scale = {
            "vector_data": drawings.get('lines', []) + drawings.get('curves', []),
            "texts": page.get('texts', [])
        }
        
        # Step 2: Call Scale API
        logger.info("Calling Scale API")
        scale_response = requests.post(SCALE_API_URL, json=input_for_scale)
        
        if scale_response.status_code != 200:
            raise HTTPException(
                status_code=scale_response.status_code,
                detail=f"Scale API error: {scale_response.text}"
            )
        
        scale_data = scale_response.json()
        logger.info("Scale API response received")
        
        # Combine results
        result = {
            "vector_data": vector_data,
            "scale_data": scale_data
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
