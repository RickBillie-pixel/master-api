import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Master API",
    description="Processes PDF by calling Vector Drawing API and Pre-Filter API",
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
PRE_FILTER_API_URL = "https://pre-filter-scale-api.onrender.com/pre-filter/"

@app.post("/process/")
async def process_pdf(file: UploadFile):
    """Process PDF: Extract vectors via Vector Drawing API, save to JSON, then filter via Pre-Filter API"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Step 1: Call Vector Drawing API with specified parameters
        file_content = await file.read()
        files = {'file': (file.filename, file_content, 'application/pdf')}
        params = {
            'minify': 'false',  # Use non-minified output
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
        
        # Get the raw response text
        raw_response = vector_response.text
        logger.info(f"Raw Vector Drawing API response (first 100 chars): {raw_response[:100]}")
        logger.info(f"Response length: {len(raw_response)} bytes")
        
        # Don't use requests.json() - explicitly parse the JSON string
        try:
            # Use standard json module to parse
            vector_data = json.loads(raw_response)
            logger.info(f"JSON parsed successfully, type={type(vector_data)}")
            logger.info(f"Top-level keys: {list(vector_data.keys()) if isinstance(vector_data, dict) else 'Not a dict'}")
            
            # Double-check if we need to parse again (in case of double-encoding)
            if isinstance(vector_data, str):
                logger.warning("Parsed result is still a string, attempting to parse again")
                vector_data = json.loads(vector_data)
                logger.info(f"Second parse: type={type(vector_data)}")
                logger.info(f"Second parse keys: {list(vector_data.keys()) if isinstance(vector_data, dict) else 'Not a dict'}")
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            # Save problematic response for debugging
            debug_path = f"/tmp/vector_response_{uuid.uuid4()}.json"
            with open(debug_path, 'w') as f:
                f.write(raw_response)
            logger.info(f"Saved problematic response to {debug_path}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Vector Drawing API response: {str(e)}")
        
        # Ensure we have the expected structure
        if not isinstance(vector_data, dict):
            raise HTTPException(status_code=400, detail="Parsed result is not a dictionary")
            
        if 'pages' not in vector_data or not vector_data['pages']:
            raise HTTPException(status_code=400, detail="No pages found in vector data")
            
        # Get the first page
        page = vector_data['pages'][0]
        
        # Extract drawings and texts
        drawings = page.get('drawings', {})
        texts = page.get('texts', [])
        
        logger.info(f"Found {len(texts)} texts and drawing types: {list(drawings.keys())}")
        
        # Prepare data for Pre-Filter API
        vector_data_for_filter = {
            "vector_data": [],
            "texts": []
        }
        
        # Process lines from drawings
        if 'lines' in drawings and isinstance(drawings['lines'], list):
            for line in drawings['lines']:
                if 'p1' in line and 'p2' in line:
                    vector_data_for_filter["vector_data"].append({
                        "type": line.get("type", "line"),
                        "p1": line["p1"],
                        "p2": line["p2"],
                        "length": line.get("length")
                    })
        
        # Process texts
        for text in texts:
            if 'text' in text and 'position' in text:
                vector_data_for_filter["texts"].append({
                    "text": text["text"],
                    "position": text["position"]
                })
        
        logger.info(f"Prepared {len(vector_data_for_filter['vector_data'])} vectors and {len(vector_data_for_filter['texts'])} texts for Pre-Filter API")
        
        # Save to temporary file
        temp_file_path = f"/tmp/filter_input_{uuid.uuid4()}.json"
        with open(temp_file_path, 'w') as f:
            json.dump(vector_data_for_filter, f)
        
        logger.info(f"Saved input for Pre-Filter API to {temp_file_path}")
        
        # Call Pre-Filter API
        try:
            with open(temp_file_path, 'rb') as f:
                filter_files = {'file': (os.path.basename(temp_file_path), f, 'application/json')}
                logger.info("Calling Pre-Filter API")
                filter_response = requests.post(PRE_FILTER_API_URL, files=filter_files, timeout=300)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            if filter_response.status_code != 200:
                logger.warning(f"Pre-Filter API error: {filter_response.status_code}")
                return {
                    "vector_data": vector_data,
                    "filtered_data": None,
                    "error": filter_response.text
                }
            
            # Parse Pre-Filter API response
            try:
                filtered_data = filter_response.json()
                logger.info("Pre-Filter API response parsed successfully")
            except Exception as e:
                logger.error(f"Error parsing Pre-Filter API response: {e}")
                return {
                    "vector_data": vector_data,
                    "filtered_data": None,
                    "error": f"Error parsing Pre-Filter API response: {str(e)}"
                }
            
            # Return combined results
            return {
                "vector_data": vector_data,
                "filtered_data": filtered_data,
                "timestamp": "2025-07-18"
            }
            
        except Exception as e:
            logger.error(f"Error calling Pre-Filter API: {e}")
            return {
                "vector_data": vector_data,
                "filtered_data": None,
                "error": f"Error calling Pre-Filter API: {str(e)}"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
