import fitz  # PyMuPDF
import requests
import json
from typing import Dict, List
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return temporary file path."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
        return tmp_file.name
    except requests.RequestException as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        raise

def extract_vector_data(pdf_path: str, page_number: int) -> Dict:
    """Extract vector data (lines, texts) from a specific PDF page."""
    try:
        doc = fitz.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Invalid page number: {page_number}")
        
        page = doc.load_page(page_number)
        page_width, page_height = page.rect.width, page.rect.height

        # Extract drawings (vector data)
        drawings = page.get_drawings()
        lines = []
        texts = []

        for drawing in drawings:
            for item in drawing["items"]:
                if item[0] == "l":  # Line
                    x0, y0, x1, y1 = item[1]
                    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                    lines.append({
                        "type": "line",
                        "p1": {"x": x0, "y": y0},
                        "p2": {"x": x1, "y": y1},
                        "length": length,
                        "color": item[2][:3] if len(item[2]) >= 3 else [0, 0, 0],
                        "width": item[2][3] if len(item[2]) > 3 else 0.0
                    })

        # Extract text
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        texts.append({
                            "text": span["text"].strip(),
                            "position": {"x": span["bbox"][0], "y": span["bbox"][1]},
                            "font_size": span["size"],
                            "font_name": span["font"],
                            "color": span["color"],
                            "bbox": {
                                "x0": span["bbox"][0],
                                "y0": span["bbox"][1],
                                "x1": span["bbox"][2],
                                "y1": span["bbox"][3],
                                "width": span["bbox"][2] - span["bbox"][0],
                                "height": span["bbox"][3] - span["bbox"][1]
                            }
                        })

        return {
            "page_number": page_number + 1,
            "page_size": {"width": page_width, "height": page_height},
            "drawings": {"lines": lines, "texts": texts},
            "is_vector": bool(lines),
            "processing_time_ms": 0  # Placeholder
        }

    except Exception as e:
        logger.error(f"Error extracting vector data: {str(e)}")
        return {"error": str(e), "page_number": page_number + 1}

def extract_scale_data(pdf_path: str, page_number: int) -> Dict:
    """Extract scale data from text on a specific PDF page."""
    try:
        doc = fitz.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Invalid page number: {page_number}")
        
        page = doc.load_page(page_number)
        text = page.get_text("text")

        # Simple scale detection (e.g., "1:50", "1:20")
        scale_ratios = {"1:50": 0.02, "1:20": 0.05}  # m/pixel approximations
        detected_scale = None
        confidence = 0.0

        for ratio, scale in scale_ratios.items():
            if ratio in text:
                detected_scale = scale
                confidence = 0.95
                break

        if not detected_scale:
            detected_scale = 0.02  # Default to 1:50
            confidence = 0.5

        return {
            "scale": detected_scale,
            "scale_ratio": [k for k, v in scale_ratios.items() if v == detected_scale][0],
            "real_meters_per_drawn_cm": detected_scale * 100,
            "method": "text_extraction" if confidence > 0.7 else "inference",
            "unit": "m",
            "message": f"Scale found in {('text_extraction' if confidence > 0.7 else 'inference')}: {scale_ratios}",
            "confidence": confidence,
            "validation": {"status": True, "reason": "Text reliable" if confidence > 0.7 else "Inferred"},
            "page_number": page_number + 1,
            "version": "2025-07"
        }

    except Exception as e:
        logger.error(f"Error extracting scale data: {str(e)}")
        return {"error": str(e), "page_number": page_number + 1}

def process_drawing(input_data: List[Dict]) -> Dict:
    """Process input data with URL and page number to extract vector and scale data."""
    result = {}
    for item in input_data:
        url = item.get("url")
        page_number = item.get("page_number", 0) - 1  # Adjust to 0-based indexing

        try:
            # Download PDF
            pdf_path = download_pdf_from_url(url)
            vector_data = extract_vector_data(pdf_path, page_number)
            scale_data = extract_scale_data(pdf_path, page_number)

            # Clean up temporary file
            os.unlink(pdf_path)

            if "error" in vector_data or "error" in scale_data:
                result[item] = {
                    "message": "Drawing processing failed",
                    "vector_data": vector_data,
                    "scale_data": scale_data,
                    "processing_time_ms": 0
                }
            else:
                result[item] = {
                    "vector_data": vector_data,
                    "scale_data": [scale_data],
                    "message": "Drawing processed successfully",
                    "processing_time_ms": 0  # Replace with actual timing
                }

        except Exception as e:
            result[item] = {"error": str(e), "message": "Processing failed"}

    return result

if __name__ == "__main__":
    # Example input
    input_data = [
        {
            "url": "https://pdf-temp-files.s3.us-west-2.amazonaws.com/QB2NJN6ZKSEI2ZJCOKGFJ7NJ3QTJK6XY/result_page1.pdf?X-Amz-Expires=3600&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIZJDPLX6D7EHVCKA/20250713/us-west-2/s3/aws4_request&X-Amz-Date=20250713T154741Z&X-Amz-SignedHeaders=host&X-Amz-Signature=71d42c389b4d7bd2291b08b4bb7470677661381ed21937f4a9e0aacba47d16ff",
            "page_number": 1
        }
    ]
    result = process_drawing(input_data)
    print(json.dumps(result, indent=2))
