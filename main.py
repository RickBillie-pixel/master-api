import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import logging
import requests
import json
import time
import asyncio
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API with Scale Integration",
    description="Processes PDF with Vector Drawing API, Filter API, and Scale API",
    version="4.0.0"
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
FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/filter-from-vector-api/"
SCALE_API_URL = "https://scale-api-69gl.onrender.com/calculate-scale/"  

MAX_RETRIES = 3

async def call_vector_api_with_retry(file_content: bytes, filename: str, params: dict) -> requests.Response:
    """Call Vector Drawing API with retry logic"""
    
    logger.info(f"=== Vector API Call ===")
    logger.info(f"File: {filename} ({len(file_content)} bytes)")
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            
            files = {'file': (filename, file_content, 'application/pdf')}
            headers = {
                'User-Agent': 'Master-API/4.0.0',
                'Accept': 'application/json',
                'Connection': 'close'
            }
            
            response = requests.post(
                VECTOR_API_URL,
                files=files,
                params=params,
                headers=headers,
                timeout=(30, 600)
            )
            
            if response.status_code == 200:
                logger.info("✅ Vector API call successful")
                return response
            else:
                logger.error(f"❌ Vector API returned {response.status_code}")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Vector API failed after {MAX_RETRIES} attempts"
                    )
                    
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=504, detail="Vector API timeout")
                
        except Exception as e:
            logger.error(f"💥 Error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail=f"Vector API failed: {str(e)}")
        
        if attempt < MAX_RETRIES - 1:
            wait_time = 2 ** attempt + 1
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    raise HTTPException(status_code=500, detail="Vector API failed after all retries")

def convert_vision_coordinates_to_pdf(vision_data: Dict, pdf_page_size: Dict) -> Dict:
    """Convert vision coordinates from image pixels to PDF coordinates"""
    if not vision_data or not pdf_page_size:
        return vision_data
    
    try:
        image_meta = vision_data.get("image_metadata", {})
        image_width_px = image_meta.get("image_width_pixels", 1)
        image_height_px = image_meta.get("image_height_pixels", 1)
        
        pdf_width_pts = pdf_page_size.get("width", 595)
        pdf_height_pts = pdf_page_size.get("height", 842)
        
        scale_x = pdf_width_pts / image_width_px
        scale_y = pdf_height_pts / image_height_px
        
        converted_regions = []
        for region in vision_data.get("regions", []):
            coord_block = region.get("coordinate_block", [])
            if len(coord_block) >= 4:
                x1_img, y1_img, x2_img, y2_img = coord_block
                
                x1_pdf = x1_img * scale_x
                y1_pdf = y1_img * scale_y
                x2_pdf = x2_img * scale_x
                y2_pdf = y2_img * scale_y
                
                converted_region = region.copy()
                converted_region["coordinate_block"] = [
                    round(x1_pdf, 1), 
                    round(y1_pdf, 1), 
                    round(x2_pdf, 1), 
                    round(y2_pdf, 1)
                ]
                converted_regions.append(converted_region)
        
        converted_vision = vision_data.copy()
        converted_vision["regions"] = converted_regions
        
        logger.info(f"✅ Converted {len(converted_regions)} regions")
        return converted_vision
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return vision_data

async def call_filter_api(vector_data: Dict, converted_vision: Dict, debug: bool) -> Dict:
    """Call Filter API with retry logic"""
    logger.info("=== Calling Filter API ===")
    
    for attempt in range(MAX_RETRIES):
        try:
            filter_url = f"{FILTER_API_URL}?debug={str(debug).lower()}"
            filter_data = {
                "vector_data": vector_data,
                "vision_output": converted_vision
            }
            
            filter_response = requests.post(
                filter_url,
                json=filter_data,
                headers={'Content-Type': 'application/json'},
                timeout=300
            )
            
            if filter_response.status_code == 200:
                logger.info("✅ Filter API call successful")
                return filter_response.json()
            else:
                logger.error(f"❌ Filter API returned {filter_response.status_code}: {filter_response.text}")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(status_code=500, detail="Filter API failed")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Filter API request failed on attempt {attempt + 1}: {e}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail="Filter API connection failed")
        
        if attempt < MAX_RETRIES - 1:
            wait_time = 2 ** attempt + 1
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    raise HTTPException(status_code=500, detail="Filter API failed after all retries")

async def call_scale_api(filtered_data: Dict, debug: bool) -> Dict:
    """Call Scale API with retry logic"""
    logger.info("=== Calling Scale API ===")
    
    for attempt in range(MAX_RETRIES):
        try:
            scale_url = f"{SCALE_API_URL}?debug={str(debug).lower()}"
            
            scale_response = requests.post(
                scale_url,
                json=filtered_data,
                headers={'Content-Type': 'application/json'},
                timeout=300
            )
            
            if scale_response.status_code == 200:
                logger.info("✅ Scale API call successful")
                return scale_response.json()
            else:
                logger.error(f"❌ Scale API returned {scale_response.status_code}: {scale_response.text}")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(status_code=500, detail="Scale API failed")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Scale API request failed on attempt {attempt + 1}: {e}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail="Scale API connection failed")
        
        if attempt < MAX_RETRIES - 1:
            wait_time = 2 ** attempt + 1
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    raise HTTPException(status_code=500, detail="Scale API failed after all retries")

@app.post("/process/")
async def process_pdf_with_scale(
    file: UploadFile = File(...),
    vision_output: str = Form(...),
    output_format: str = Form(default="json"),
    debug: bool = Form(default=False),
    minify: bool = Form(default=True)
):
    """Process PDF with Vector API, Filter API, and Scale API integration"""
    try:
        logger.info(f"=== Starting PDF Processing with Scale Integration ===")
        logger.info(f"File: {file.filename}, Debug: {debug}, Minify: {minify}")
        
        # Read file
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Parse vision output
        try:
            vision_data = json.loads(vision_output)
            regions_count = len(vision_data.get('regions', []))
            logger.info(f"Vision regions: {regions_count}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")

        # Call Vector API
        logger.info("=== Calling Vector API ===")
        
        params = {
            'minify': str(minify).lower(),
            'remove_non_essential': 'false',
            'precision': '1'
        }
        
        vector_response = await call_vector_api_with_retry(file_content, file.filename, params)
        
        # Parse vector response
        try:
            vector_data = json.loads(vector_response.text)
            if isinstance(vector_data, str):
                vector_data = json.loads(vector_data)
        except Exception as e:
            logger.error(f"Vector API parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector API response")

        # Extract PDF page dimensions and convert coordinates
        logger.info("=== Converting Coordinates ===")
        
        first_page = vector_data['pages'][0]
        pdf_page_size = first_page.get('page_size', {"width": 595.0, "height": 842.0})
        converted_vision = convert_vision_coordinates_to_pdf(vision_data, pdf_page_size)

        # Call Filter API
        filtered_data = await call_filter_api(vector_data, converted_vision, debug)
        
        # Call Scale API with filtered data
        scale_data = await call_scale_api(filtered_data, debug)
        
        # Create combined result
        result = {
            "status": "success",
            "filter_output": filtered_data,
            "scale_output": scale_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_info": {
                "vector_api_lines": len(vector_data.get('pages', [{}])[0].get('drawings', {}).get('lines', [])),
                "vector_api_texts": len(vector_data.get('pages', [{}])[0].get('texts', [])),
                "filtered_regions": len(filtered_data.get('regions', [])),
                "total_filtered_lines": sum(len(r.get('lines', [])) for r in filtered_data.get('regions', [])),
                "total_filtered_texts": sum(len(r.get('texts', [])) for r in filtered_data.get('regions', [])),
                "scale_regions": len(scale_data.get('regions', [])),
                "overall_scale_pt_per_mm": scale_data.get('overall_average_pt_per_mm'),
                "minified": minify
            }
        }
        
        # Apply minification if requested
        if minify:
            logger.info("Applying minification...")
            
            # Minify filter output
            minified_filter = {
                "drawing_type": filtered_data.get("drawing_type"),
                "regions": []
            }
            
            for region in filtered_data.get("regions", []):
                minified_region = {
                    "label": region.get("label"),
                    "lines": [
                        {
                            "length": line.get("length"),
                            "orientation": line.get("orientation"),
                            "midpoint": line.get("midpoint")
                        } for line in region.get("lines", [])
                    ],
                    "texts": [
                        {
                            "text": text.get("text"),
                            "midpoint": text.get("midpoint"),
                            "orientation": text.get("orientation")
                        } for text in region.get("texts", [])
                    ]
                }
                minified_filter["regions"].append(minified_region)
            
            # Minify scale output
            minified_scale = {
                "drawing_type": scale_data.get("drawing_type"),
                "overall_average_pt_per_mm": scale_data.get("overall_average_pt_per_mm"),
                "overall_average_mm_per_pt": scale_data.get("overall_average_mm_per_pt"),
                "regions": []
            }
            
            for region in scale_data.get("regions", []):
                minified_scale_region = {
                    "region_label": region.get("region_label"),
                    "average_scale_pt_per_mm": region.get("average_scale_pt_per_mm"),
                    "average_scale_mm_per_pt": region.get("average_scale_mm_per_pt"),
                    "confidence": region.get("confidence"),
                    "validation_status": region.get("validation_status")
                }
                minified_scale["regions"].append(minified_scale_region)
            
            result["filter_output"] = minified_filter
            result["scale_output"] = minified_scale
        
        # Log summary
        logger.info(f"✅ Processing complete:")
        logger.info(f"  Drawing type: {filtered_data.get('drawing_type')}")
        logger.info(f"  Regions: {len(filtered_data.get('regions', []))}")
        logger.info(f"  Total lines: {result['processing_info']['total_filtered_lines']}")
        logger.info(f"  Total texts: {result['processing_info']['total_filtered_texts']}")
        logger.info(f"  Overall scale: {scale_data.get('overall_average_pt_per_mm', 'N/A')} pt/mm")
        
        # Return appropriate format
        if output_format == "txt":
            summary = f"""=== PDF PROCESSING RESULT ===
Status: {result['status']}
Drawing Type: {filtered_data.get('drawing_type')}
Regions: {len(filtered_data.get('regions', []))}
Total Lines: {result['processing_info']['total_filtered_lines']}
Total Texts: {result['processing_info']['total_filtered_texts']}
Overall Scale: {scale_data.get('overall_average_pt_per_mm', 'N/A')} pt/mm
Timestamp: {result['timestamp']}
Minified: {minify}

FILTER REGIONS:
"""
            for region in filtered_data.get('regions', []):
                summary += f"- {region.get('label')}: {len(region.get('lines', []))} lines, {len(region.get('texts', []))} texts\n"
            
            summary += "\nSCALE REGIONS:\n"
            for region in scale_data.get('regions', []):
                summary += f"- {region.get('region_label')}: {region.get('average_scale_pt_per_mm', 'N/A')} pt/mm (confidence: {region.get('confidence', 0)}%)\n"
            
            return PlainTextResponse(content=summary)
        else:
            return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "4.0.0",
        "description": "Master API with Scale Integration"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Master API with Scale Integration",
        "version": "4.0.0",
        "description": "Processes PDF with Vector Drawing API, Filter API, and Scale API",
        "workflow": [
            "1. Receive PDF file and vision_output (region coordinates)",
            "2. Call Vector Drawing API to extract lines and texts", 
            "3. Convert vision coordinates from image pixels to PDF points",
            "4. Call Filter API to filter and organize data per region",
            "5. Call Scale API to calculate scales from filtered data",
            "6. Return both Filter output and Scale output"
        ],
        "apis_integrated": {
            "vector_api": "https://vector-drawning.onrender.com/extract/",
            "filter_api": "https://pre-filter-scale-api.onrender.com/filter-from-vector-api/",
            "scale_api": "https://scale-api.onrender.com/calculate-scale/"
        },
        "output_structure": {
            "filter_output": "Filtered lines and texts per region with orientations and midpoints",
            "scale_output": "Calculated scales per region with confidence scores and validation"
        },
        "features": [
            "Coordinate conversion from image pixels to PDF points",
            "Region-based filtering with geometric intersection",
            "Scale calculation with dimension-to-line matching",
            "Minification support for cleaner output",
            "Retry logic for all API calls",
            "Combined Filter + Scale output"
        ],
        "endpoints": {
            "/process/": "Main processing endpoint (supports minify=true/false)",
            "/health/": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Master API with Scale Integration v4.0.0 on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
