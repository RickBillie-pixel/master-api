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
        
        # Parse the JSON response
        raw_response = vector_response.text
        logger.info(f"Raw Vector Drawing API response (first 1000 chars): {raw_response[:1000]}")
        logger.info(f"Response length: {len(raw_response)} bytes")
        
        # CRITICAL FIX: Explicitly use json.loads to parse the string
        try:
            vector_data = json.loads(raw_response)
            if isinstance(vector_data, str):
                # If it's still a string, try parsing it again
                logger.warning("First parse resulted in a string, trying again")
                vector_data = json.loads(vector_data)
                
            logger.info(f"Successfully parsed response, type={type(vector_data)}")
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            
            # Try to extract information from a malformed JSON as a last resort
            # Write raw response to a temporary file for debugging
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as debug_file:
                debug_file.write(raw_response)
                debug_path = debug_file.name
            logger.info(f"Wrote raw response to {debug_path}")
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse Vector Drawing API response: {str(e)}"
            )
        
        logger.info("Vector Drawing API response processed successfully")
        
        # Debug output of the parsed data structure
        logger.info(f"Parsed data type: {type(vector_data)}")
        logger.info(f"Parsed data keys: {vector_data.keys() if isinstance(vector_data, dict) else 'Not a dict'}")
        
        # Verify we have the expected structure
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
