from fastapi import FastAPI
from app.routers import detect_image_video
from contextlib import asynccontextmanager
from app.services.load_trained_model import load_model
# from app.services.load_trained_model import lifespan
from fastapi.middleware.cors import CORSMiddleware
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clean up on shutdown"""
    # Startup
    logger.info("Starting up FastAPI application...")
    try:
        logger.info("Loading model on startup...")
        model = load_model()
        if model is not None:
            logger.info("✅ Model loaded successfully during startup!")
        else:
            logger.error("❌ Failed to load model during startup")
    except Exception as e:
        logger.error(f"❌ Error loading model during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    
    
app = FastAPI(
    title="Face Mask Detection API",
    description="YOLO-based face mask detection using Hugging Face model",
    version="1.0.0",
    lifespan=lifespan
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
    return {
        "message": "Face Mask Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect_image": "POST /detect/image - Upload image for detection",
            "detect_video": "POST /detect/video - Upload video for detection",
        }
    }

# ✅ Register routers AFTER CORS middleware
app.include_router(detect_image_video.router)