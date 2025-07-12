"""
Master API - Coordinates Vector Extraction, Scale Detection, and Wall Detection APIs
Optimized by Grok 4 Heavy for production-readiness and school project testing
"""

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import asyncio
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("master_api")

app = FastAPI(
    title="Master API",
    description="Coordinates Vector Extraction, Scale Detection, and Wall Detection for construction drawings",
    version="1.0.0",
    docs_url="/docs/",
    openapi_url="/openapi.json"
)

app_start_time = datetime.now()

# Custom exceptions
class MasterProcessingError(HTTPException):
    """Exception for Master API processing errors"""
    def __init__(self, detail: str = "Processing error"):
        super().__init__(status_code=400, detail=detail)

class MasterAPIError(HTTPException):
    """Exception for downstream API failures"""
    def __init__(self, detail: str = "Downstream API error"):
        super().__init__(status_code=500, detail=detail)

# Models
class DrawingRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of the PDF drawing")
    page_number: int = Field(..., ge=1, description="Page number to process")

class MasterResponse(BaseModel):
    vector_data: Dict[str, Any] = Field(..., description="Extracted vector data")
    scale_data: List[Dict[str, Any]] = Field(..., description="Detected scale data")
    wall_data: Dict[str, Any] = Field(..., description="Detected wall data")
    message: str = Field(..., description="Processing status")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")

# Configuration
class MasterConfig:
    def __init__(self):
        self.vector_api_url = os.getenv("VECTOR_API_URL", "https://vector-api-0wlf.onrender.com/extract-vectors-from-urls/")
        self.scale_api_url = os.getenv("SCALE_API_URL", "https://your-scale-api.onrender.com/detect-scale-from-json/")
        self.wall_api_url = os.getenv("WALL_API_URL", "https://wall-api.onrender.com/detect-walls/")
        self.request_timeout = float(os.getenv("REQUEST_TIMEOUT", "30.0"))

config = MasterConfig()

# Metrics
request_count = 0
error_count = 0
total_processing_time = 0

@app.middleware("http")
async def add_request_id_and_metrics(request, call_next):
    global request_count, total_processing_time, error_count
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    request_count += 1
    total_processing_time += duration
    if response.status_code >= 400:
        error_count += 1
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

@app.get("/metrics/")
async def get_metrics():
    return {
        "request_count": request_count,
        "error_count": error_count,
        "average_response_time_ms": total_processing_time / request_count if request_count > 0 else 0,
        "uptime_seconds": (datetime.now() - app_start_time).total_seconds()
    }

@app.get("/")
async def root():
    """Root endpoint with API overview"""
    return {
        "service": "Master API",
        "version": "1.0.0",
        "description": "Coordinates Vector Extraction, Scale Detection, and Wall Detection for construction drawings",
        "endpoints": {
            "process_drawing": "/process-drawing/",
            "health": "/health/",
            "metrics": "/metrics/",
            "docs": "/docs/"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint with diagnostics"""
    memory = get_memory_usage()
    return {
        "status": "healthy",
        "service": "master-api",
        "version": "1.0.0",
        "memory_usage_mb": round(memory, 2),
        "timestamp": datetime.now().isoformat(),
        "config": vars(config)
    }

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except (ImportError, AttributeError):
        logger.warning("psutil not available or memory_info failed, memory monitoring disabled")
        return 0.0

@app.post("/process-drawing/", response_model=MasterResponse)
async def process_drawing(request: DrawingRequest, background_tasks: BackgroundTasks):
    """Process a drawing by coordinating Vector Extraction, Scale Detection, and Wall Detection APIs"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"Processing drawing: {request.url}, page {request.page_number} [Request ID: {request_id}]")

    try:
        # Step 1: Call Vector Extraction API
        vector_response = requests.post(
            config.vector_api_url,
            json=[{"url": str(request.url), "page_number": request.page_number}],
            headers={"Content-Type": "application/json"},
            timeout=config.request_timeout
        )
        vector_response.raise_for_status()
        vector_data = vector_response.json()
        if not vector_data or not isinstance(vector_data, list) or not vector_data[0].get("data"):
            raise MasterProcessingError("Invalid vector data received")
        logger.info(f"Vector data extracted: {len(vector_data[0]['data']['drawings']['lines'])} lines, "
                   f"{len(vector_data[0]['data']['texts'])} texts")

        # Step 2: Call Scale Detection API
        scale_response = requests.post(
            config.scale_api_url,
            json={"pages": [vector_data[0]["data"]], "summary": vector_data[0]["data"]},
            headers={"Content-Type": "application/json"},
            timeout=config.request_timeout
        )
        scale_response.raise_for_status()
        scale_data = scale_response.json()
        if not scale_data or scale_data[0].get("confidence", 0) == 0:
            logger.warning("No reliable scale found")
        else:
            logger.info(f"Scale detected: {scale_data[0]['scale_ratio']} (confidence: {scale_data[0]['confidence']})")

        # Step 3: Call Wall Detection API
        wall_response = requests.post(
            config.wall_api_url,
            json={"vector_data": vector_data[0]["data"], "scale_data": scale_data[0]},
            headers={"Content-Type": "application/json"},
            timeout=config.request_timeout
        )
        wall_response.raise_for_status()
        wall_data = wall_response.json()
        logger.info(f"Wall data received: {wall_data.get('message', 'No message')}")

        processing_time_ms = int((time.time() - start_time) * 1000)
        return MasterResponse(
            vector_data=vector_data[0]["data"],
            scale_data=scale_data,
            wall_data=wall_data,
            message="Drawing processed successfully",
            processing_time_ms=processing_time_ms
        )

    except requests.RequestException as e:
        logger.error(f"Error calling downstream API: {e}", exc_info=True)
        raise MasterAPIError(f"Failed to call downstream API: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing drawing: {e}", exc_info=True)
        raise MasterProcessingError(f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)