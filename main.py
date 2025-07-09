"""
Master API - PDF URL Analysis
Takes a PDF URL and page number, orchestrates all microservices
Returns structured data with page number for workflow merging
"""
from fastapi import FastAPI, HTTPException
import httpx
import requests
import logging
from typing import Dict, Any, List
import json
import io
from datetime import datetime
from pydantic import BaseModel, Field
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_api")

app = FastAPI(
    title="PDF Expert Master API",
    description="Analyzes PDFs from URLs with page number tracking for workflow merging",
    version="1.2.0",
)

# Service URLs - Configure with environment variables to make deployment flexible
# Default to localhost for development, but allow override with environment variables
SERVICES = {
    "vector": os.environ.get("VECTOR_API_URL", "https://vector-api-1zqj.onrender.com"),
    "view_type": os.environ.get("VIEW_TYPE_API_URL", "https://view-type-api.onrender.com"), 
    "wall": os.environ.get("WALL_API_URL", "https://wall-api.onrender.com"),
    "room": os.environ.get("ROOM_API_URL", "https://room-api-adwe.onrender.com"),
    "component": os.environ.get("COMPONENT_API_URL", "https://component-api.onrender.com"),
    "installation": os.environ.get("INSTALLATION_API_URL", "https://installation-api.onrender.com"),
    "filter": os.environ.get("FILTER_API_URL", "https://filter-api-babk.onrender.com")
}

# Verify at startup that all API endpoints are correctly configured
for service_name, service_url in SERVICES.items():
    logger.info(f"Configured {service_name} API at: {service_url}")

class PDFURLRequest(BaseModel):
    """Request model for PDF URL processing"""
    pdf_url: str = Field(..., description="URL of the PDF to analyze")
    page_number: int = Field(1, description="Page number to analyze (1-based)")

@app.post("/analyze-pdf-url/")
async def analyze_pdf_url(request: PDFURLRequest):
    """
    Analyze a PDF from a URL with page number tracking
    
    Args:
        request: PDFURLRequest containing PDF URL and page number
        
    Returns:
        Structured JSON with complete analysis results including page number for workflow merging
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
        
        # Create filename from URL
        filename = request.pdf_url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename = f"document_{request.page_number}.pdf"
        
        # Create UploadFile-like object from bytes
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Step 1: Extract vectors
        logger.info("Step 1: Extracting vectors...")
        vector_data = await _call_vector_api(pdf_file, filename)
        
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
        
        # Create structured result for workflow merging
        result = _create_structured_result(
            filename,
            vector_data,
            view_type_data,
            wall_data,
            room_data,
            component_data,
            installation_data,
            filtered_data
        )
        
        # Add page information to result for workflow merging
        result["page_number"] = request.page_number
        result["source"] = "url"
        result["pdf_url"] = request.pdf_url
        
        logger.info(f"PDF URL analysis completed successfully for page {request.page_number}")
        return result
        
    except Exception as e:
        logger.error(f"Error during PDF URL analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def _call_vector_api(pdf_file: io.BytesIO, filename: str) -> Dict[str, Any]:
    """Call vector extraction API with PDF data"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:  # 15 minutes for large PDFs
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            files = {"file": (filename, pdf_file.read(), "application/pdf")}
            api_url = f"{SERVICES['vector']}/extract-vectors/"
            logger.info(f"Calling Vector API at: {api_url}")
            
            response = await client.post(api_url, files=files)
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
            api_url = f"{SERVICES['view_type']}/detect-view-type/"
            logger.info(f"Calling View Type API at: {api_url}")
            
            response = await client.post(api_url, json=vector_data)
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
            api_url = f"{SERVICES['wall']}/detect-walls/"
            logger.info(f"Calling Wall API at: {api_url}")
            
            response = await client.post(api_url, json=request_data)
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
            api_url = f"{SERVICES['room']}/detect-rooms/"
            logger.info(f"Calling Room API at: {api_url}")
            
            response = await client.post(api_url, json=request_data)
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
            api_url = f"{SERVICES['component']}/detect-components/"
            logger.info(f"Calling Component API at: {api_url}")
            
            response = await client.post(api_url, json=request_data)
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
            api_url = f"{SERVICES['installation']}/detect-installations/"
            logger.info(f"Calling Installation API at: {api_url}")
            
            response = await client.post(api_url, json=request_data)
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
            api_url = f"{SERVICES['filter']}/filter-data/"
            logger.info(f"Calling Filter API at: {api_url}")
            
            response = await client.post(api_url, json=request_data)
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

def _create_structured_result(filename: str, vector_data: Dict[str, Any], 
                            view_type_data: Dict[str, Any], wall_data: Dict[str, Any],
                            room_data: Dict[str, Any], component_data: Dict[str, Any],
                            installation_data: Dict[str, Any], filtered_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create structured result for workflow merging
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
        "analysis_status": "completed",
        "timestamp": datetime.now().isoformat()
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
            "unlinked_texts": page.get("unlinked_texts", []),
            "errors": page.get("errors", [])
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
        "view_type": view_type_data.get("pages", [{}])[0].get("view_type", "unknown") if len(view_type_data.get("pages", [])) > 0 else "unknown",
        "analysis_timestamp": datetime.now().isoformat(),
        "services_used": list(SERVICES.keys()),
        "api_version": "1.2.0"
    }
    
    return result

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PDF Expert Master API",
        "version": "1.2.0",
        "description": "Analyzes PDFs from URLs with page number tracking for workflow merging",
        "endpoints": {
            "analyze_pdf_url": "/analyze-pdf-url/",
            "health": "/health/"
        },
        "services": {name: url for name, url in SERVICES.items()}
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    # Check connection to each service
    service_status = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for name, url in SERVICES.items():
            try:
                # Try to connect to the health endpoint of each service
                health_url = f"{url}/health/"
                response = await client.get(health_url)
                if response.status_code == 200:
                    service_status[name] = "available"
                else:
                    service_status[name] = f"error: {response.status_code}"
            except Exception as e:
                service_status[name] = f"error: {str(e)}"
    
    return {
        "status": "healthy", 
        "service": "master-api",
        "timestamp": datetime.now().isoformat(),
        "services": service_status
    }

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)