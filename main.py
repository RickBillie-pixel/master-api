"""
Master API - Orchestrates PDF Processing
Downloads PDF pages from URLs, extracts vectors, and detects scales
"""

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("master_api")

app = FastAPI(
    title="Master PDF Processing API",
    description="Orchestrates PDF download, vector extraction, and scale detection",
    version="1.0.0"
)

class PageInput(BaseModel):
    url: str
    page_number: int

class ProcessResponse(BaseModel):
    scales: List[dict]
    message: str
    timestamp: str

VECTOR_API_URL = "https://vector-api-0wlf.onrender.com/extract-vectors/"
SCALE_API_URL = "https://scale-api-5f65.onrender.com/detect-scale-from-json/"

@app.post("/process-pdf/", response_model=ProcessResponse)
async def process_pdf(inputs: List[PageInput]):
    """
    Process PDF pages:
    1. Download each page PDF from URL
    2. Extract vectors for each page
    3. Combine into single vector data
    4. Send to scale detection
    """
    if not inputs:
        raise HTTPException(status_code=400, detail="No input pages provided")
    
    logger.info(f"Processing {len(inputs)} PDF pages")
    
    pages = []
    total_file_size = 0
    total_processing_time = 0
    
    # Sort inputs by page number
    sorted_inputs = sorted(inputs, key=lambda x: x.page_number)
    
    for inp in sorted_inputs:
        # Download PDF
        try:
            download_resp = requests.get(inp.url, timeout=30)
            download_resp.raise_for_status()
            pdf_content = download_resp.content
            if len(pdf_content) == 0:
                raise ValueError("Empty PDF content")
            total_file_size += len(pdf_content)
        except Exception as e:
            logger.error(f"Failed to download PDF from {inp.url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to download PDF page {inp.page_number}: {str(e)}")
        
        # Extract vectors
        try:
            files = {"file": ("page.pdf", pdf_content, "application/pdf")}
            vector_resp = requests.post(VECTOR_API_URL, files=files, timeout=60)
            vector_resp.raise_for_status()
            vector_data = vector_resp.json()
            
            if not vector_data.get("pages"):
                raise ValueError("No pages in vector response")
            
            page_data = vector_data["pages"][0]
            page_data["page_number"] = inp.page_number
            pages.append(page_data)
            
            total_processing_time += vector_data["summary"].get("processing_time_ms", 0)
        except Exception as e:
            logger.error(f"Vector extraction failed for page {inp.page_number}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vector extraction failed for page {inp.page_number}: {str(e)}")
    
    # Combine into VectorData
    combined_summary = {
        "total_pages": len(pages),
        "total_lines": sum(len(p.get("drawings", {}).get("lines", [])) for p in pages),
        "total_rectangles": sum(len(p.get("drawings", {}).get("rectangles", [])) for p in pages),
        "total_curves": sum(len(p.get("drawings", {}).get("curves", [])) for p in pages),
        "total_texts": sum(len(p.get("texts", [])) for p in pages),
        "file_size_mb": round(total_file_size / (1024 * 1024), 2),
        "processing_time_ms": total_processing_time
    }
    
    vector_full = {
        "pages": pages,
        "summary": combined_summary
    }
    
    # Detect scale
    try:
        scale_resp = requests.post(SCALE_API_URL, json=vector_full, timeout=60)
        scale_resp.raise_for_status()
        scale_data = scale_resp.json()
    except Exception as e:
        logger.error(f"Scale detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scale detection failed: {str(e)}")
    
    return ProcessResponse(
        scales=scale_data,
        message="Processing completed successfully",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "master-api",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Master PDF Processing API",
        "version": "1.0.0",
        "description": "Downloads PDF pages, extracts vectors, and detects scales",
        "endpoints": {
            "process_pdf": "/process-pdf/",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
