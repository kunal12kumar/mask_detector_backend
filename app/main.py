from fastapi import FastAPI
from app.routers import detect_image_video
from app.services.load_trained_model import lifespan
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Face Mask Detection API",
    description="YOLO-based face mask detection using Hugging Face model",
    version="1.0.0",
    # lifespan=lifespan
)

# ✅ Add CORS middleware FIRST, before registering routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Face Mask Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Upload image for detection",
            "health": "GET /health - Check API health",
            "model-info": "GET /model-info - Get model details"
        }
    }

# ✅ Register routers AFTER CORS middleware
app.include_router(detect_image_video.router)