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
    title="Master API v4.1.1 - Compatible with Fixed APIs",
    description="Processes PDF with Vector Drawing API, Filter API v7.0.1, and Scale API v7.0.1",
    version="4.1.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# UPDATED API URLs to point to the FIXED APIs
VECTOR_API_URL = "https://vector-drawning.onrender.com/extract/"
FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/filter-from-vector-api/"  # Your fixed Filter API
SCALE_API_URL = "https://scale-api-69gl.onrender.com/calculate-scale/"  # Your fixed Scale API

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
                'User-Agent': 'Master-API/4.1.1',
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
        
        logger.info(f"Coordinate conversion: {image_width_px}x{image_height_px}px → {pdf_width_pts}x{pdf_height_pts}pts")
        logger.info(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        
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
                
                logger.debug(f"Region '{region.get('label')}': [{x1_img},{y1_img},{x2_img},{y2_img}]px → [{x1_pdf:.1f},{y1_pdf:.1f},{x2_pdf:.1f},{y2_pdf:.1f}]pts")
        
        converted_vision = vision_data.copy()
        converted_vision["regions"] = converted_regions
        
        logger.info(f"✅ Converted {len(converted_regions)} regions from image to PDF coordinates")
        return converted_vision
        
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return vision_data

async def call_filter_api(vector_data: Dict, converted_vision: Dict, debug: bool) -> Dict:
    """Call Filter API v7.0.1 with retry logic"""
    logger.info("=== Calling Filter API v7.0.1 ===")
    
    for attempt in range(MAX_RETRIES):
        try:
            filter_url = f"{FILTER_API_URL}?debug={str(debug).lower()}"
            filter_data = {
                "vector_data": vector_data,
                "vision_output": converted_vision
            }
            
            logger.info(f"Filter API request: {len(vector_data.get('pages', []))} pages, {len(converted_vision.get('regions', []))} regions")
            
            filter_response = requests.post(
                filter_url,
                json=filter_data,
                headers={'Content-Type': 'application/json'},
                timeout=300
            )
            
            if filter_response.status_code == 200:
                logger.info("✅ Filter API call successful")
                filter_result = filter_response.json()
                
                # Log filter results
                total_lines = sum(len(r.get('lines', [])) for r in filter_result.get('regions', []))
                total_texts = sum(len(r.get('texts', [])) for r in filter_result.get('regions', []))
                logger.info(f"Filter output: {len(filter_result.get('regions', []))} regions, {total_lines} lines, {total_texts} texts")
                
                return filter_result
            else:
                logger.error(f"❌ Filter API returned {filter_response.status_code}: {filter_response.text}")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(status_code=500, detail=f"Filter API failed: {filter_response.text}")
                    
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
    """Call Scale API v7.0.1 with retry logic"""
    logger.info("=== Calling Scale API v7.0.1 ===")
    
    for attempt in range(MAX_RETRIES):
        try:
            # Log what we're sending to Scale API
            total_regions = len(filtered_data.get('regions', []))
            logger.info(f"Scale API request: {total_regions} regions")
            for region in filtered_data.get('regions', []):
                lines_count = len(region.get('lines', []))
                texts_count = len(region.get('texts', []))
                parsed_type = region.get('parsed_drawing_type', 'none')
                logger.debug(f"  Region '{region.get('label')}': {lines_count} lines, {texts_count} texts, parsed_type: {parsed_type}")
            
            scale_response = requests.post(
                SCALE_API_URL,
                json=filtered_data,
                headers={'Content-Type': 'application/json'},
                timeout=300
            )
            
            if scale_response.status_code == 200:
                logger.info("✅ Scale API call successful")
                scale_result = scale_response.json()
                
                # Log scale results
                total_calculations = scale_result.get('total_calculations', 0)
                global_avg = scale_result.get('global_average_scale_pt_per_mm', 'N/A')
                logger.info(f"Scale output: {total_calculations} total calculations, global avg: {global_avg} pt/mm")
                
                return scale_result
            else:
                logger.error(f"❌ Scale API returned {scale_response.status_code}: {scale_response.text}")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(status_code=500, detail=f"Scale API failed: {scale_response.text}")
                    
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
    """Process PDF with Vector API, Filter API v7.0.1, and Scale API v7.0.1 integration"""
    try:
        logger.info(f"=== Starting PDF Processing with Scale Integration v4.1.1 ===")
        logger.info(f"File: {file.filename}, Debug: {debug}, Minify: {minify}")
        
        # Read file
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Parse vision output
        try:
            vision_data = json.loads(vision_output)
            regions_count = len(vision_data.get('regions', []))
            drawing_type = vision_data.get('drawing_type', 'unknown')
            logger.info(f"Vision input: {drawing_type} with {regions_count} regions")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid vision_output JSON: {str(e)}")

        # Call Vector API
        logger.info("=== Step 1: Vector API ===")
        
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
                
            # Log vector extraction results
            first_page = vector_data.get('pages', [{}])[0]
            lines_count = len(first_page.get('drawings', {}).get('lines', []))
            texts_count = len(first_page.get('texts', []))
            logger.info(f"Vector extraction: {lines_count} lines, {texts_count} texts")
            
        except Exception as e:
            logger.error(f"Vector API parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector API response: {str(e)}")

        # Convert coordinates from image pixels to PDF points
        logger.info("=== Step 2: Coordinate Conversion ===")
        
        first_page = vector_data['pages'][0]
        pdf_page_size = first_page.get('page_size', {"width": 595.0, "height": 842.0})
        converted_vision = convert_vision_coordinates_to_pdf(vision_data, pdf_page_size)

        # Call Filter API v7.0.1
        logger.info("=== Step 3: Filter API v7.0.1 ===")
        filtered_data = await call_filter_api(vector_data, converted_vision, debug)
        
        # Call Scale API v7.0.1 with filtered data
        logger.info("=== Step 4: Scale API v7.0.1 ===")
        scale_data = await call_scale_api(filtered_data, debug)
        
        # Create combined result
        result = {
            "status": "success",
            "filter_output": filtered_data,
            "scale_output": scale_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_info": {
                "drawing_type": filtered_data.get("drawing_type"),
                "vector_api_lines": len(vector_data.get('pages', [{}])[0].get('drawings', {}).get('lines', [])),
                "vector_api_texts": len(vector_data.get('pages', [{}])[0].get('texts', [])),
                "filtered_regions": len(filtered_data.get('regions', [])),
                "total_filtered_lines": sum(len(r.get('lines', [])) for r in filtered_data.get('regions', [])),
                "total_filtered_texts": sum(len(r.get('texts', [])) for r in filtered_data.get('regions', [])),
                "scale_regions": len(scale_data.get('regions', [])),
                "total_scale_calculations": scale_data.get('total_calculations', 0),
                "global_average_scale": scale_data.get('global_average_scale_pt_per_mm'),
                "minified": minify,
                "api_versions": {
                    "filter_api": "7.0.1-fixed",
                    "scale_api": "7.0.1-fixed",
                    "master_api": "4.1.1-fixed"
                }
            }
        }
        
        # Apply minification if requested
        if minify:
            logger.info("Applying minification...")
            
            # Minify filter output (UNCHANGED - Filter API v7.0.1 format is already clean)
            minified_filter = {
                "drawing_type": filtered_data.get("drawing_type"),
                "regions": []
            }
            
            for region in filtered_data.get("regions", []):
                minified_region = {
                    "label": region.get("label"),
                    "parsed_drawing_type": region.get("parsed_drawing_type"),  # Include this for Scale API
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
                            "midpoint": text.get("midpoint")
                        } for text in region.get("texts", [])
                    ]
                }
                minified_filter["regions"].append(minified_region)
            
            # Minify scale output - UPDATED FOR SCALE API v7.0.1
            minified_scale = {
                "drawing_type": scale_data.get("drawing_type"),
                "total_regions": scale_data.get("total_regions"),
                "total_calculations": scale_data.get("total_calculations"),
                "global_average_scale_pt_per_mm": scale_data.get("global_average_scale_pt_per_mm"),
                "global_average_scale_mm_per_pt": scale_data.get("global_average_scale_mm_per_pt"),
                "global_average_formula": scale_data.get("global_average_formula"),
                "regions": []
            }
            
            for region in scale_data.get("regions", []):
                minified_scale_region = {
                    "region_label": region.get("region_label"),
                    "drawing_type": region.get("drawing_type"),
                    "parsed_drawing_type": region.get("parsed_drawing_type"),  # Include parsed type
                    "dimension_strategy": region.get("dimension_strategy"),
                    "horizontal_calculations": region.get("horizontal_calculations", []),
                    "vertical_calculations": region.get("vertical_calculations", []),
                    "total_calculations": region.get("total_calculations", 0),
                    "average_scale_pt_per_mm": region.get("average_scale_pt_per_mm"),
                    "average_scale_mm_per_pt": region.get("average_scale_mm_per_pt"),
                    "average_calculation_formula": region.get("average_calculation_formula")
                }
                minified_scale["regions"].append(minified_scale_region)
            
            result["filter_output"] = minified_filter
            result["scale_output"] = minified_scale
        
        # Log comprehensive summary
        logger.info(f"✅ Processing complete:")
        logger.info(f"  Drawing type: {filtered_data.get('drawing_type')}")
        logger.info(f"  Regions processed: {len(filtered_data.get('regions', []))}")
        logger.info(f"  Total filtered lines: {result['processing_info']['total_filtered_lines']}")
        logger.info(f"  Total filtered texts: {result['processing_info']['total_filtered_texts']}")
        logger.info(f"  Total scale calculations: {scale_data.get('total_calculations', 0)}")
        logger.info(f"  Global average scale: {scale_data.get('global_average_scale_pt_per_mm', 'N/A')} pt/mm")
        
        # Return appropriate format
        if output_format == "txt":
            summary = f"""=== PDF PROCESSING RESULT v4.1.1 ===
Status: {result['status']}
Drawing Type: {filtered_data.get('drawing_type')}
Regions: {len(filtered_data.get('regions', []))}
Total Lines: {result['processing_info']['total_filtered_lines']}
Total Texts: {result['processing_info']['total_filtered_texts']}
Total Scale Calculations: {scale_data.get('total_calculations', 0)}
Global Average Scale: {scale_data.get('global_average_scale_pt_per_mm', 'N/A')} pt/mm
Timestamp: {result['timestamp']}
Minified: {minify}

FILTER REGIONS:
"""
            for region in filtered_data.get('regions', []):
                parsed_type = region.get('parsed_drawing_type', 'none')
                summary += f"- {region.get('label')}: {len(region.get('lines', []))} lines, {len(region.get('texts', []))} texts, type: {parsed_type}\n"
            
            summary += "\nSCALE REGIONS:\n"
            for region in scale_data.get('regions', []):
                h_count = len(region.get('horizontal_calculations', []))
                v_count = len(region.get('vertical_calculations', []))
                total_calcs = region.get('total_calculations', 0)
                avg_scale = region.get('average_scale_pt_per_mm', 'N/A')
                summary += f"- {region.get('region_label')}: {total_calcs} calculations ({h_count}H + {v_count}V), avg scale: {avg_scale} pt/mm\n"
            
            if scale_data.get('global_average_formula'):
                summary += f"\nGLOBAL AVERAGE CALCULATION:\n{scale_data.get('global_average_formula')}\n"
            
            return PlainTextResponse(content=summary)
        else:
            return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf_with_scale: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "4.1.1",
        "description": "Master API with Fixed Scale Integration",
        "api_compatibility": {
            "filter_api": "7.0.1-fixed (Pydantic compatible)",
            "scale_api": "7.0.1-fixed (Pydantic v2 compatible)",
            "vector_api": "latest"
        },
        "bug_fixes": [
            "✅ Fixed Pydantic v2 compatibility issue",
            "✅ Updated to use fixed Filter and Scale APIs",
            "✅ Enhanced logging for parsed_drawing_type support",
            "✅ Proper error handling for JSON string fields"
        ],
        "features": [
            "Compatible with Filter API v7.0.1 fixed rules",
            "Compatible with Scale API v7.0.1 (Pydantic v2)",
            "Coordinate conversion from image pixels to PDF points",
            "Comprehensive logging and error handling",
            "Minification with correct field names"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Master API v4.1.1 - Fixed Scale API Compatible",
        "version": "4.1.1",
        "description": "FIXED: Processes PDF with Vector API, Filter API v7.0.1, and Scale API v7.0.1",
        "bug_fix": {
            "issue": "Scale API v7.0.0 Pydantic validation error on dimension_mapping field",
            "root_cause": "Dict objects being passed to Pydantic v2 model expecting strings",
            "solution": "Updated to use fixed Scale API v7.0.1 with proper JSON string serialization"
        },
        "compatibility": {
            "filter_api": "v7.0.1-fixed (correct filtering logic + Pydantic compatibility)",
            "scale_api": "v7.0.1-fixed (Pydantic v2 compatible with JSON string fields)",
            "vector_api": "latest (PDF vector extraction)"
        },
        "workflow": [
            "1. Receive PDF file and vision_output (region coordinates in image pixels)",
            "2. Call Vector Drawing API to extract lines and texts from PDF", 
            "3. Convert vision coordinates from image pixels to PDF points",
            "4. Call Filter API v7.0.1 to filter and organize data per region (fixed filtering logic)",
            "5. Call Scale API v7.0.1 to calculate scales with proper Pydantic v2 compatibility",
            "6. Return both Filter output and Scale output with comprehensive stats"
        ],
        "apis_integrated": {
            "vector_api": "https://vector-drawning.onrender.com/extract/",
            "filter_api": "https://pre-filter-scale-api.onrender.com/filter-from-vector-api/ (v7.0.1-fixed)",
            "scale_api": "https://scale-api-69gl.onrender.com/calculate-scale/ (v7.0.1-fixed)"
        },
        "output_structure": {
            "filter_output": {
                "description": "Filtered lines and texts per region",
                "features": ["Orientations (H/V/diagonal)", "Midpoints", "25pt region buffer", "parsed_drawing_type field"]
            },
            "scale_output": {
                "description": "Scale calculations per region",
                "features": ["3 horizontal + 3 vertical per region", "Calculation formulas", "Regional and global averages", "Physical dimension mapping"]
            },
            "processing_info": {
                "description": "Comprehensive statistics",
                "includes": ["API versions", "Counts per stage", "Global averages", "Parsed drawing types"]
            }
        },
        "improvements_v4_1_1": [
            "✅ Fixed Pydantic v2 compatibility issue in Scale API integration",
            "✅ Updated to use corrected Filter API v7.0.1 with proper filtering logic",
            "✅ Enhanced logging for bestektekening region parsing",
            "✅ Proper handling of parsed_drawing_type field",
            "✅ Comprehensive error handling and debugging"
        ],
        "pydantic_compatibility": {
            "filter_api": "Pydantic v1.10.18",
            "scale_api": "Pydantic v2.6.4 (fixed with JSON string serialization)",
            "master_api": "No Pydantic models (pure FastAPI)"
        },
        "endpoints": {
            "/process/": "Main processing endpoint (supports minify=true/false, output_format=json/txt)",
            "/health/": "Health check with API compatibility info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Master API v4.1.1 - Fixed Scale API Compatible on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
