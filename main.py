# master.py (Rewritten to call Vector API for extraction, Scale API for scale, batch lines, send batches to Wall API, then validate)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
from typing import List, Dict, Any
import time
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("master_api")

app = FastAPI(
    title="Master API",
    description="Orchestrates PDF processing for wall detection",
    version="2025-07",
)

# Service URLs (set as env vars or constants)
VECTOR_API_URL = "https://vector-api-0wlf.onrender.com/extract-vector/"  
SCALE_API_URL = "https://scale-api-5f65.onrender.com/detect-scale/"  
WALL_DETECTION_URL = "https://wall-api.onrender.com/detect-walls/"
VALIDATION_URL = "https://validation-api-21ha.onrender.com/validate-walls/"
BATCH_SIZE = 5000  # Batch size for lines

class ProcessRequest(BaseModel):
    pdf_url: str
    page_number: int = 1

@app.post("/process-drawing/")
async def process_drawing(request: ProcessRequest):
    request_id = str(uuid.uuid4())
    try:
        logger.info(f"Processing drawing: {request.pdf_url}, page {request.page_number} [Request ID: {request_id}]")
        
        # Step 1: Call Vector API to extract lines and texts
        vector_response = requests.post(
            VECTOR_API_URL,
            json={"pdf_url": request.pdf_url, "page_number": request.page_number},
            timeout=300
        )
        vector_response.raise_for_status()
        vector_data = vector_response.json()
        lines = vector_data.get("lines", [])  # Assuming response has "lines" and "texts"
        texts = vector_data.get("texts", [])
        logger.info(f"Vector data extracted: {len(lines)} lines, {len(texts)} texts")
        
        # Step 2: Call Scale API to detect scale
        scale_response = requests.post(
            SCALE_API_URL,
            json={"texts": texts},
            timeout=300
        )
        scale_response.raise_for_status()
        scale_info = scale_response.json()
        scale_m_per_pixel = scale_info["scale_m_per_pixel"]  # Assuming key in response
        logger.info(f"Scale detected: {scale_info.get('scale', 'unknown')} (confidence: {scale_info.get('confidence', 0)})")
        
        # Step 3: Batch lines and call Wall Detection API for each batch
        batches = [lines[i:i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]
        all_walls = []
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch)} lines")
            wall_response = requests.post(
                WALL_DETECTION_URL,
                json={"indexed_lines": batch, "scale_m_per_pixel": scale_m_per_pixel, "texts": texts},
                timeout=600
            )
            wall_response.raise_for_status()
            batch_walls = wall_response.json()
            all_walls.extend(batch_walls)
        logger.info(f"Detected {len(all_walls)} walls across all batches")
        
        # Step 4: Call Validation API with all walls
        validation_response = requests.post(
            VALIDATION_URL,
            json={"walls": all_walls},
            timeout=300
        )
        validation_response.raise_for_status()
        validated_walls = validation_response.json()["validated_walls"]
        logger.info(f"Validation completed")
        
        return {"walls": validated_walls, "request_id": request_id}
    except Exception as e:
        logger.error(f"Error processing drawing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Master API", "version": "2025-07"}

@app.get("/health/")
async def health_check():
    return {"status": "healthy", "service": "master_api", "version": "2025-07", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    from gunicorn.app.base import BaseApplication
    import os

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.application = app
            self.options = options or {}
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    port = int(os.environ.get("PORT", 8000))
    options = {
        "bind": f"0.0.0.0:{port}",
        "workers": 4,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "timeout": 300
    }
    StandaloneApplication(app, options).run()
