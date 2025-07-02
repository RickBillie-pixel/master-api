"""
Master API - Single entry point for PDF analysis
Takes a PDF file and internally orchestrates all microservices
Returns structured data for n8n integration
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
import httpx
import logging
from typing import Dict, Any, List
import json

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

async def _call_vector_api(file: UploadFile) -> Dict[str, Any]:
    """Call vector extraction API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        try:
            files = {"file": (file.filename, await file.read(), "application/pdf")}
            response = await client.post(f"{SERVICES['vector']}/extract-vectors/", files=files)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.error(f"Timeout calling vector API after 5 minutes")
            raise HTTPException(status_code=408, detail="Vector extraction timed out - PDF may be too large or complex")
        except httpx.HTTPStatusError as e:
            logger.error(f"Vector API returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Vector API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Error calling vector API: {e}")
            raise HTTPException(status_code=500, detail=f"Vector API error: {str(e)}")

async def _call_view_type_api(vector_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call view type detection API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        try:
            response = await client.post(f"{SERVICES['view_type']}/detect-view-type/", json=vector_data)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.error(f"Timeout calling view type API")
            raise HTTPException(status_code=408, detail="View type detection timed out")
        except Exception as e:
            logger.error(f"Error calling view type API: {e}")
            raise HTTPException(status_code=500, detail=f"View type API error: {str(e)}")

async def _call_wall_api(vector_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call wall detection API"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "pages": vector_data["pages"],
            "scale_m_per_pixel": 1.0  # Default scale, could be calculated
        }
        response = await client.post(f"{SERVICES['wall']}/detect-walls/", json=request_data)
        response.raise_for_status()
        return response.json()

async def _call_room_api(vector_data: Dict[str, Any], wall_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call room detection API"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "pages": vector_data["pages"],
            "walls": wall_data["pages"],
            "scale_m_per_pixel": 1.0
        }
        response = await client.post(f"{SERVICES['room']}/detect-rooms/", json=request_data)
        response.raise_for_status()
        return response.json()

async def _call_component_api(vector_data: Dict[str, Any], wall_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call component detection API"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "pages": vector_data["pages"],
            "walls": wall_data["pages"],
            "scale_m_per_pixel": 1.0
        }
        response = await client.post(f"{SERVICES['component']}/detect-components/", json=request_data)
        response.raise_for_status()
        return response.json()

async def _call_installation_api(vector_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call installation detection API"""
    async with httpx.AsyncClient() as client:
        request_data = {"pages": vector_data["pages"]}
        response = await client.post(f"{SERVICES['installation']}/detect-installations/", json=request_data)
        response.raise_for_status()
        return response.json()

async def _call_filter_api(vector_data: Dict[str, Any], wall_data: Dict[str, Any], 
                          room_data: Dict[str, Any], component_data: Dict[str, Any], 
                          installation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call filter API"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "pages": vector_data["pages"],
            "walls": wall_data["pages"],
            "rooms": room_data["pages"],
            "components": component_data["pages"],
            "symbols": installation_data["pages"],
            "scale_m_per_pixel": 1.0
        }
        response = await client.post(f"{SERVICES['filter']}/filter-data/", json=request_data)
        response.raise_for_status()
        return response.json()

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

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "master-api"}

@app.get("/services/")
async def list_services():
    """List all available services"""
    return {"services": SERVICES}

@app.get("/services/health/")
async def check_all_services():
    """Check health of all microservices"""
    async with httpx.AsyncClient() as client:
        health_status = {}
        for service_name, url in SERVICES.items():
            try:
                response = await client.get(f"{url}/health/", timeout=10)
                health_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "status_code": response.status_code,
                    "url": url
                }
            except Exception as e:
                health_status[service_name] = {
                    "status": "error",
                    "error": str(e),
                    "url": url
                }
        return {"services_health": health_status}

@app.get("/config/")
async def get_config():
    """Get current configuration"""
    return {
        "services": SERVICES,
        "version": "1.0.0",
        "description": "PDF Expert Master API - Orchestrates all microservices"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 