"""
Master API - Single entry point for PDF analysis
Takes a PDF file and internally orchestrates all microservices
Returns structured data for n8n integration
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
import httpx
import requests
import logging
from typing import Dict, Any, List
import json
import base64
import io
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_api")

app = FastAPI(
    title="PDF Expert Master API",
    description="Single entry point for complete PDF analysis - orchestrates all microservices",
    version="1.0.0",
)

# Service URLs - All services are now live on Render.com
SERVICES = {
    "vector": "https://vector-api-1zqj.onrender.com",
    "view_type": "https://view-type-api.onrender.com", 
    "wall": "https://wall-api.onrender.com",
    "room": "https://room-api-adwe.onrender.com",
    "component": "https://component-api.onrender.com",
    "installation": "https://installation-api.onrender.com",
    "filter": "https://filter-api-babk.onrender.com"
}

class Base64PDFRequest(BaseModel):
    """Request model for base64-encoded PDF data"""
    base64_pdf: str
    filename: str = "page.pdf"
    page_number: int = 1
    
    class Config:
        # Add this for compatibility with Pydantic v1
        extra = "forbid"

class PDFURLRequest(BaseModel):
    """Request model for PDF URL processing"""
    pdf_url: str
    filename: str = "document.pdf"
    page_number: int = 1
    
    class Config:
        # Add this for compatibility with Pydantic v1
        extra = "forbid"

@app.post("/analyze-pdf/")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Complete PDF analysis - single endpoint that orchestrates all services
    
    Args:
        file: PDF file to analyze
        
    Returns:
        Structured JSON with complete analysis results for n8n
    """
    try:
        logger.info(f"Starting complete analysis of {file.filename}")
        
        # Step 1: Extract vectors
        logger.info("Step 1: Extracting vectors...")
        vector_data = await _call_vector_api(file)
        
        # Step 2: Detect view type
        logger.info("Step 2: Detecting view type...")
        view_type_data = await _call_view_type_api(vector_data)
        
        # Step 3: Detect walls
        logger.info("Step 3: Detecting walls...")
        wall_data = await _call_wall_api(vector_data)
        
        # Step 4: Detect rooms
        logger.info("Step 4: Detecting rooms...")
        room_data = await _call_room_api(vector_data, wall_data)
        
        # Step 5: Detect components
        logger.info("Step 5: Detecting components...")
        component_data = await _call_component_api(vector_data, wall_data)
        
        # Step 6: Detect installations
        logger.info("Step 6: Detecting installations...")
        installation_data = await _call_installation_api(vector_data)
        
        # Step 7: Filter data
        logger.info("Step 7: Filtering data...")
        filtered_data = await _call_filter_api(
            vector_data, wall_data, room_data, component_data, installation_data
        )
        
        # Create structured result for n8n
        result = _create_structured_result(
            file.filename,
            vector_data,
            view_type_data,
            wall_data,
            room_data,
            component_data,
            installation_data,
            filtered_data
        )
        
        logger.info("Analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-base64-pdf/")
async def analyze_base64_pdf(request: Base64PDFRequest):
    """
    Complete PDF analysis for base64-encoded PDF data
    
    Args:
        request: Base64PDFRequest containing base64 PDF data and metadata
        
    Returns:
        Structured JSON with complete analysis results for n8n
    """
    try:
        logger.info(f"Starting analysis of base64 PDF: {request.filename} (page {request.page_number})")
        
        # Decode base64 to PDF bytes
        try:
            pdf_bytes = base64.b64decode(request.base64_pdf)
        except Exception as e:
            logger.error(f"Invalid base64 data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 PDF data")
        
        # Create UploadFile-like object from bytes
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Step 1: Extract vectors
        logger.info("Step 1: Extracting vectors...")
        vector_data = await _call_vector_api_base64(pdf_file, request.filename)
        
        # Step 2: Detect view type
        logger.info("Step 2: Detecting view type...")
        view_type_data = await _call_view_type_api(vector_data)
        
        # Step 3: Detect walls
        logger.info("Step 3: Detecting walls...")
        wall_data = await _call_wall_api(vector_data)
        
        # Step 4: Detect rooms
        logger.info("Step 4: Detecting rooms...")
        room_data = await _call_room_api(vector_data, wall_data)
        
        # Step 5: Detect components
        logger.info("Step 5: Detecting components...")
        component_data = await _call_component_api(vector_data, wall_data)
        
        # Step 6: Detect installations
        logger.info("Step 6: Detecting installations...")
        installation_data = await _call_installation_api(vector_data)
        
        # Step 7: Filter data
        logger.info("Step 7: Filtering data...")
        filtered_data = await _call_filter_api(
            vector_data, wall_data, room_data, component_data, installation_data
        )
        
        # Create structured result for n8n
        result = _create_structured_result(
            request.filename,
            vector_data,
            view_type_data,
            wall_data,
            room_data,
            component_data,
            installation_data,
            filtered_data
        )
        
        # Add page information to result
        result["page_number"] = request.page_number
        result["source"] = "base64"
        
        logger.info("Base64 PDF analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error during base64 PDF analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-pdf-url/")
async def analyze_pdf_url(request: PDFURLRequest):
    """
    Complete PDF analysis for PDF from URL
    
    Args:
        request: PDFURLRequest containing PDF URL and metadata
        
    Returns:
        Structured JSON with complete analysis results for n8n
    """
    try:
        logger.info(f"Starting analysis of PDF from URL: {request.pdf_url} (page {request.page_number})")
        
        # Download PDF from URL
        try:
            logger.info(f"Downloading PDF from URL: {request.pdf_url}")
            response = requests.get(request.pdf_url, timeout=300)  # 5 minutes timeout for download
            response.raise_for_status()
            pdf_bytes = response.content
            logger.info(f"Successfully downloaded PDF: {len(pdf_bytes)} bytes")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDF from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {str(e)}")
        
        # Create UploadFile-like object from bytes
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Step 1: Extract vectors
        logger.info("Step 1: Extracting vectors...")
        vector_data = await _call_vector_api_base64(pdf_file, request.filename)
        
        # Step 2: Detect view type
        logger.info("Step 2: Detecting view type...")
        view_type_data = await _call_view_type_api(vector_data)
        
        # Step 3: Detect walls
        logger.info("Step 3: Detecting walls...")
        wall_data = await _call_wall_api(vector_data)
        
        # Step 4: Detect rooms
        logger.info("Step 4: Detecting rooms...")
        room_data = await _call_room_api(vector_data, wall_data)
        
        # Step 5: Detect components
        logger.info("Step 5: Detecting components...")
        component_data = await _call_component_api(vector_data, wall_data)
        
        # Step 6: Detect installations
        logger.info("Step 6: Detecting installations...")
        installation_data = await _call_installation_api(vector_data)
        
        # Step 7: Filter data
        logger.info("Step 7: Filtering data...")
        filtered_data = await _call_filter_api(
            vector_data, wall_data, room_data, component_data, installation_data
        )
        
        # Create structured result for n8n
        result = _create_structured_result(
            request.filename,
            vector_data,
            view_type_data,
            wall_data,
            room_data,
            component_data,
            installation_data,
            filtered_data
        )
        
        # Add page information to result
        result["page_number"] = request.page_number
        result["source"] = "url"
        result["pdf_url"] = request.pdf_url
        
        logger.info("PDF URL analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error during PDF URL analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def _call_vector_api(file: UploadFile) -> Dict[str, Any]:
    """Call vector extraction API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:  # 15 minutes for large PDFs
        try:
            files = {"file": (file.filename, await file.read(), "application/pdf")}
            logger.info(f"Uploading {file.filename} to vector API...")
            response = await client.post(f"{SERVICES['vector']}/extract-vectors/", files=files)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Vector extraction completed: {result.get('summary', {}).get('total_pages', 0)} pages")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling vector API after 15 minutes")
            raise HTTPException(status_code=408, detail="Vector extraction timed out - PDF may be too large or complex")
        except httpx.HTTPStatusError as e:
            logger.error(f"Vector API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Vector API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling vector API: {e}")
            raise HTTPException(status_code=500, detail=f"Vector API error: {str(e)}")

async def _call_view_type_api(vector_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call view type detection API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:  # 2 minutes for large PDFs
        try:
            logger.info(f"Processing {len(vector_data['pages'])} pages for view type detection...")
            response = await client.post(f"{SERVICES['view_type']}/detect-view-type/", json=vector_data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"View type detection completed")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling view type API after 2 minutes")
            raise HTTPException(status_code=408, detail="View type detection timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"View type API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"View type API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling view type API: {e}")
            raise HTTPException(status_code=500, detail=f"View type API error: {str(e)}")

async def _call_wall_api(vector_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call wall detection API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:  # 10 minutes for large PDFs
        try:
            request_data = {
                "pages": vector_data["pages"],
                "scale_m_per_pixel": 1.0  # Default scale, could be calculated
            }
            logger.info(f"Processing {len(vector_data['pages'])} pages for wall detection...")
            response = await client.post(f"{SERVICES['wall']}/detect-walls/", json=request_data)
            response.raise_for_status()
            result = response.json()
            total_walls = sum(len(page.get('walls', [])) for page in result.get('pages', []))
            logger.info(f"Wall detection completed: {total_walls} walls found")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling wall API after 10 minutes")
            raise HTTPException(status_code=408, detail="Wall detection timed out - PDF may be too large or complex")
        except httpx.HTTPStatusError as e:
            logger.error(f"Wall API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Wall API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling wall API: {e}")
            raise HTTPException(status_code=500, detail=f"Wall API error: {str(e)}")

async def _call_room_api(vector_data: Dict[str, Any], wall_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call room detection API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:  # 5 minutes for large PDFs
        try:
            request_data = {
                "pages": vector_data["pages"],
                "walls": wall_data["pages"],
                "scale_m_per_pixel": 1.0
            }
            logger.info(f"Processing {len(vector_data['pages'])} pages for room detection...")
            response = await client.post(f"{SERVICES['room']}/detect-rooms/", json=request_data)
            response.raise_for_status()
            result = response.json()
            total_rooms = sum(len(page.get('rooms', [])) for page in result.get('pages', []))
            logger.info(f"Room detection completed: {total_rooms} rooms found")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling room API after 5 minutes")
            raise HTTPException(status_code=408, detail="Room detection timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Room API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Room API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling room API: {e}")
            raise HTTPException(status_code=500, detail=f"Room API error: {str(e)}")

async def _call_component_api(vector_data: Dict[str, Any], wall_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call component detection API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:  # 5 minutes for large PDFs
        try:
            request_data = {
                "pages": vector_data["pages"],
                "walls": wall_data["pages"],
                "scale_m_per_pixel": 1.0
            }
            logger.info(f"Processing {len(vector_data['pages'])} pages for component detection...")
            response = await client.post(f"{SERVICES['component']}/detect-components/", json=request_data)
            response.raise_for_status()
            result = response.json()
            total_components = sum(len(page.get('components', [])) for page in result.get('pages', []))
            logger.info(f"Component detection completed: {total_components} components found")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling component API after 5 minutes")
            raise HTTPException(status_code=408, detail="Component detection timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Component API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Component API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling component API: {e}")
            raise HTTPException(status_code=500, detail=f"Component API error: {str(e)}")

async def _call_installation_api(vector_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call installation detection API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(240.0)) as client:  # 4 minutes for large PDFs
        try:
            request_data = {"pages": vector_data["pages"]}
            logger.info(f"Processing {len(vector_data['pages'])} pages for installation detection...")
            response = await client.post(f"{SERVICES['installation']}/detect-installations/", json=request_data)
            response.raise_for_status()
            result = response.json()
            total_installations = sum(len(page.get('symbols', [])) for page in result.get('pages', []))
            logger.info(f"Installation detection completed: {total_installations} installations found")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling installation API after 4 minutes")
            raise HTTPException(status_code=408, detail="Installation detection timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Installation API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Installation API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling installation API: {e}")
            raise HTTPException(status_code=500, detail=f"Installation API error: {str(e)}")

async def _call_filter_api(vector_data: Dict[str, Any], wall_data: Dict[str, Any], 
                          room_data: Dict[str, Any], component_data: Dict[str, Any], 
                          installation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call filter API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:  # 5 minutes for large PDFs
        try:
            request_data = {
                "pages": vector_data["pages"],
                "walls": wall_data["pages"],
                "rooms": room_data["pages"],
                "components": component_data["pages"],
                "symbols": installation_data["pages"],
                "scale_m_per_pixel": 1.0
            }
            logger.info(f"Processing {len(vector_data['pages'])} pages for data filtering...")
            response = await client.post(f"{SERVICES['filter']}/filter-data/", json=request_data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Data filtering completed successfully")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling filter API after 5 minutes")
            raise HTTPException(status_code=408, detail="Data filtering timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Filter API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Filter API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling filter API: {e}")
            raise HTTPException(status_code=500, detail=f"Filter API error: {str(e)}")

async def _call_vector_api_base64(pdf_file: io.BytesIO, filename: str) -> Dict[str, Any]:
    """Call vector extraction API with base64-decoded PDF data"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:  # 15 minutes for large PDFs
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            files = {"file": (filename, pdf_file.read(), "application/pdf")}
            logger.info(f"Uploading {filename} to vector API...")
            response = await client.post(f"{SERVICES['vector']}/extract-vectors/", files=files)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Vector extraction completed: {result.get('summary', {}).get('total_pages', 0)} pages")
            return result
        except httpx.TimeoutException:
            logger.error(f"Timeout calling vector API after 15 minutes")
            raise HTTPException(status_code=408, detail="Vector extraction timed out - PDF may be too large or complex")
        except httpx.HTTPStatusError as e:
            logger.error(f"Vector API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Vector API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling vector API: {e}")
            raise HTTPException(status_code=500, detail=f"Vector API error: {str(e)}")

def _create_structured_result(filename: str, vector_data: Dict[str, Any], 
                            view_type_data: Dict[str, Any], wall_data: Dict[str, Any],
                            room_data: Dict[str, Any], component_data: Dict[str, Any],
                            installation_data: Dict[str, Any], filtered_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create structured result for n8n integration
    """
    
    # Extract final results from filtered data
    final_pages = filtered_data.get("pages", [])
    
    # Create summary statistics
    summary = {
        "filename": filename,
        "total_pages": len(final_pages),
        "total_walls": 0,
        "total_rooms": 0,
        "total_components": 0,
        "total_symbols": 0,
        "total_texts": 0,
        "analysis_status": "completed"
    }
    
    # Process each page
    pages_data = []
    for page in final_pages:
        page_summary = {
            "page_number": page.get("page_number", 0),
            "walls": page.get("walls", []),
            "rooms": page.get("rooms", []),
            "components": page.get("components", []),
            "symbols": page.get("symbols", []),
            "unlinked_texts": page.get("unlinked_texts", [])
        }
        
        # Update summary counts
        summary["total_walls"] += len(page_summary["walls"])
        summary["total_rooms"] += len(page_summary["rooms"])
        summary["total_components"] += len(page_summary["components"])
        summary["total_symbols"] += len(page_summary["symbols"])
        summary["total_texts"] += len(page_summary["unlinked_texts"])
        
        pages_data.append(page_summary)
    
    # Create structured result
    result = {
        "success": True,
        "filename": filename,
        "summary": summary,
        "pages": pages_data,
        "view_type": view_type_data.get("pages", [{}])[0].get("view_type", "unknown"),
        "analysis_timestamp": "2024-01-01T00:00:00Z",  # You can add actual timestamp
        "services_used": list(SERVICES.keys()),
        "raw_data": {
            "vector_extraction": vector_data,
            "view_type_detection": view_type_data,
            "wall_detection": wall_data,
            "room_detection": room_data,
            "component_detection": component_data,
            "installation_detection": installation_data,
            "filtered_result": filtered_data
        }
    }
    
    return result

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PDF Expert Master API",
        "version": "1.0.0",
        "description": "Single entry point for complete PDF analysis - orchestrates all microservices",
        "endpoints": {
            "analyze_pdf": "/analyze-pdf/",
            "analyze_base64_pdf": "/analyze-base64-pdf/",
            "health": "/health/"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "master-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
