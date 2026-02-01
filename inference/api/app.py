"""
FastAPI application for IQA inference.
RESTful API for endoscopic image quality assessment.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.predictor import IQAPredictor
from models.deep_learning import get_model

# Create FastAPI app
app = FastAPI(
    title="Endoscopic IQA API",
    description="REST API for Endoscopic Image Quality Assessment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


class QualityResponse(BaseModel):
    """Response model for quality prediction."""
    quality_score: float
    quality_category: str
    inference_time_ms: float
    message: str = "Success"


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global predictor
    
    try:
        # Create model
        model = get_model(model_type='lightweight')
        
        # Initialize predictor
        predictor = IQAPredictor(
            model=model,
            device='cpu',  # Use CPU by default for API
            image_size=(224, 224)
        )
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "running",
        "model_loaded": predictor is not None,
        "device": predictor.device if predictor else "none"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None,
        "device": predictor.device if predictor else "none"
    }


@app.post("/predict", response_model=QualityResponse)
async def predict_quality(file: UploadFile = File(...)):
    """
    Predict image quality for uploaded file.
    
    Args:
        file: Uploaded image file
    
    Returns:
        Quality prediction response
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        score, inference_time = predictor.predict(image, return_time=True)
        category = predictor.get_quality_category(score)
        
        return {
            "quality_score": float(score),
            "quality_category": category,
            "inference_time_ms": float(inference_time),
            "message": "Success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict image quality for multiple uploaded files.
    
    Args:
        files: List of uploaded image files
    
    Returns:
        List of quality predictions
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for file in files:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            score, inference_time = predictor.predict(image, return_time=True)
            category = predictor.get_quality_category(score)
            
            results.append({
                "filename": file.filename,
                "quality_score": float(score),
                "quality_category": category,
                "inference_time_ms": float(inference_time)
            })
        
        return JSONResponse(content={
            "message": "Success",
            "results": results,
            "total_images": len(files)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Lightweight CNN",
        "input_size": predictor.image_size,
        "device": predictor.device,
        "quality_categories": {
            "excellent": "0.8 - 1.0",
            "good": "0.6 - 0.8",
            "fair": "0.4 - 0.6",
            "poor": "0.2 - 0.4",
            "bad": "0.0 - 0.2"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
