import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import json
from typing import Dict, Any, Optional

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

def parse_json_safely(json_str: str) -> Optional[Dict]:
    """Parse JSON string safely with additional error handling and debugging"""
    try:
        # Try standard JSON parsing first
        result = json.loads(json_str)
        
        # Check if result is a dictionary
        if not isinstance(result, dict):
            logger.error(f"Parsed result is not a dictionary, type={type(result)}")
            return None
            
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        
        # Try to extract valid JSON if partial
        try:
            # Look for the ending curly brace
            if json_str.count('{') > json_str.count('}'):
                # The JSON might be truncated, try to add missing closing braces
                missing_braces = json_str.count('{') - json_str.count('}')
                fixed_json = json_str + ('}' * missing_braces)
                logger.info(f"Attempting to fix truncated JSON by adding {missing_braces} closing braces")
                return json.loads(fixed_json)
        except Exception:
            pass
        
        # Try with a different JSON parser if available
        try:
            import simplejson
            logger.info("Trying with simplejson")
            return simplejson.loads(json_str)
        except (ImportError, Exception):
            pass
            
        return None

@app.post("/process/")
async def process_pdf(file: UploadFile):
    """Process PDF: Extract vectors via Vector Drawing API, save to JSON, then calculate scale via Scale API"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Call Vector Drawing API with specified parameters
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'false',  # Changed to false to avoid minification issues
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
        
        # Parse the JSON response with enhanced error handling
        raw_response = vector_response.text
        logger.info(f"Raw Vector Drawing API response (first 1000 chars): {raw_response[:1000]}")
        logger.info(f"Response length: {len(raw_response)} bytes")
        
        # Try to get response as JSON directly from requests
        try:
            vector_data = vector_response.json()
            logger.info("Successfully parsed response using requests.json()")
        except Exception as e:
            logger.warning(f"Could not parse with requests.json(): {e}")
            
            # Use our custom safe parser
            vector_data = parse_json_safely(raw_response)
            
            if not vector_data:
                # Last resort: try to extract just what we need
                logger.info("Attempting manual extraction of required data")
                try:
                    # Write raw response to temporary file for debugging
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as debug_file:
                        debug_file.write(raw_response)
                        debug_path = debug_file.name
                    logger.info(f"Wrote raw response to {debug_path}")
                    
                    # Manual extraction of pages data
                    import re
                    pages_match = re.search(r'"pages":\s*\[\s*{(.+?)}\s*\]', raw_response, re.DOTALL)
                    if pages_match:
                        page_content = pages_match.group(1)
                        logger.info(f"Extracted page content (first 100 chars): {page_content[:100]}")
                        
                        # Extract texts
                        texts_match = re.search(r'"texts":\s*\[(.+?)\]', page_content, re.DOTALL)
                        if texts_match:
                            texts_content = texts_match.group(1)
                            
                            # Try to parse individual text items
                            text_items = []
                            for text_match in re.finditer(r'{(.+?)}', texts_content, re.DOTALL):
                                try:
                                    text_item = json.loads('{' + text_match.group(1) + '}')
                                    text_items.append(text_item)
                                except:
                                    pass
                            
                            # Extract drawings
                            drawings_match = re.search(r'"drawings":\s*{(.+?)}', page_content, re.DOTALL)
                            if drawings_match:
                                drawings_content = drawings_match.group(1)
                                
                                # Manual construction of vector_data
                                vector_data = {
                                    "pages": [
                                        {
                                            "texts": text_items,
                                            "drawings": {}
                                        }
                                    ]
                                }
                                
                                # Try to extract lines
                                lines_match = re.search(r'"lines":\s*\[(.+?)\]', drawings_content, re.DOTALL)
                                if lines_match:
                                    lines_content = lines_match.group(1)
                                    
                                    # Try to parse individual line items
                                    line_items = []
                                    for line_match in re.finditer(r'{(.+?)}', lines_content, re.DOTALL):
                                        try:
                                            line_item = json.loads('{' + line_match.group(1) + '}')
                                            line_items.append(line_item)
                                        except:
                                            pass
                                    
                                    vector_data["pages"][0]["drawings"]["lines"] = line_items
                except Exception as e:
                    logger.error(f"Manual extraction failed: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not parse Vector Drawing API response: {str(e)}"
                    )
        
        logger.info("Vector Drawing API response processed successfully")
        
        # Extra validation to ensure we have the expected structure
        if vector_data is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to parse Vector Drawing API response"
            )
            
        # Debug output of the parsed data structure
        logger.info(f"Parsed data type: {type(vector_data)}")
        logger.info(f"Parsed data keys: {vector_data.keys() if isinstance(vector_data, dict) else 'Not a dict'}")
        
        # Check if pages exists in vector_data
        if not isinstance(vector_data, dict) or 'pages' not in vector_data or not vector_data['pages']:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid or missing pages in vector data. Available keys: {list(vector_data.keys()) if isinstance(vector_data, dict) else 'None'}"
            )
        
        page = vector_data['pages'][0]
        
        # Debug logging of page structure
        logger.info(f"Page keys: {page.keys() if isinstance(page, dict) else 'Not a dict'}")
        
        drawings = page.get('drawings', {})
        texts = page.get('texts', [])
        
        logger.info(f"Found {len(texts)} texts and drawings with keys: {drawings.keys() if isinstance(drawings, dict) else 'Not a dict'}")
        
        # Prepare data for Scale API
        vector_data_for_scale = {
            "vector_data": [],
            "texts": []
        }
        
        # Add lines if they exist
        if isinstance(drawings, dict) and 'lines' in drawings and isinstance(drawings['lines'], list):
            for v in drawings['lines']:
                if isinstance(v, dict) and 'type' in v and 'p1' in v and 'p2' in v:
                    vector_data_for_scale["vector_data"].append({
                        "type": v["type"],
                        "p1": v["p1"],
                        "p2": v["p2"],
                        "length": v.get("length", None)
                    })
        
        # Add curves if they exist
        if isinstance(drawings, dict) and 'curves' in drawings and isinstance(drawings['curves'], list):
            for v in drawings['curves']:
                if isinstance(v, dict) and 'type' in v and 'p1' in v and 'p2' in v:
                    vector_data_for_scale["vector_data"].append({
                        "type": v["type"],
                        "p1": v["p1"],
                        "p2": v["p2"],
                        "length": v.get("length", None)
                    })
        
        # Add texts if they exist
        if isinstance(texts, list):
            for t in texts:
                if isinstance(t, dict) and 'text' in t and 'position' in t:
                    vector_data_for_scale["texts"].append({
                        "text": t["text"],
                        "position": t["position"]
                    })
        
        logger.info(f"Preparing Scale API input with {len(vector_data_for_scale['vector_data'])} vectors and {len(vector_data_for_scale['texts'])} texts")
        
        # Check if we have enough data to proceed
        if len(vector_data_for_scale['vector_data']) == 0:
            logger.warning("No valid vector data extracted for Scale API")
            return {
                "vector_data": vector_data,
                "scale_data": None,
                "timestamp": "2025-07-18 18:03 CEST",
                "error": "No valid vector data extracted for Scale API"
            }
            
        if len(vector_data_for_scale['texts']) == 0:
            logger.warning("No valid text data extracted for Scale API")
            return {
                "vector_data": vector_data,
                "scale_data": None,
                "timestamp": "2025-07-18 18:03 CEST",
                "error": "No valid text data extracted for Scale API"
            }
        
        # Step 2: Save to temporary JSON file
        temp_file_name = f"scale_input_{uuid.uuid4()}.json"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as temp_file:
            json.dump(vector_data_for_scale, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Temporary JSON file created at: {temp_file_path}")
        
        # Step 3: Call Scale API with the JSON file
        try:
            with open(temp_file_path, 'rb') as scale_input_file:
                files = {'file': (temp_file_name, scale_input_file, 'application/json')}
                logger.info("Calling Scale API with JSON file")
                scale_response = requests.post(SCALE_API_URL, files=files, timeout=300)
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")
            
            if scale_response.status_code != 200:
                logger.warning(f"Scale API returned non-200 status: {scale_response.status_code}")
                logger.warning(f"Scale API error response: {scale_response.text[:1000]}")
                return {
                    "vector_data": vector_data,
                    "scale_data": None,
                    "timestamp": "2025-07-18 18:03 CEST",
                    "error": f"Scale API error: {scale_response.text[:500]}"
                }
            
            # Parse the Scale API response
            try:
                scale_data = scale_response.json()
                logger.info("Scale API response received successfully")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Scale API response: {e}")
                return {
                    "vector_data": vector_data,
                    "scale_data": None,
                    "timestamp": "2025-07-18 18:03 CEST",
                    "error": f"Failed to parse Scale API response: {e}"
                }
        except Exception as e:
            logger.error(f"Error calling Scale API: {e}", exc_info=True)
            
            # If Scale API fails, we can still return the vector data
            return {
                "vector_data": vector_data,
                "scale_data": None,
                "timestamp": "2025-07-18 18:03 CEST",
                "error": f"Scale API error: {str(e)}"
            }
            
        # Combine and return results
        result = {
            "vector_data": vector_data,
            "scale_data": scale_data,
            "timestamp": "2025-07-18 18:03 CEST"
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
    return {"status": "healthy", "version": "1.0", "timestamp": "2025-07-18 18:03 CEST"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
