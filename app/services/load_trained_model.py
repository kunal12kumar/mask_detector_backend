# In this we will load our model and pass the images and videos to give the result 
# a function to lad the model
import logging
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from fastapi import FastAPI

 #Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
model_loading = False

# Replace with your Hugging Face model repository
HF_REPO_ID = "kunal12kumardev/Face_mask_detection"  # 🔄 CHANGE THIS
MODEL_FILENAME = "best.pt"

async def load_model():
    """Load model from Hugging Face Hub"""
    global model, model_loading
    
    if model is not None or model_loading:
        return
        
    model_loading = True
    logger.info("Starting model download from Hugging Face...")
    
    try:
        # Download model with cache
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir="./model_cache",  # Persistent cache directory
            force_download=False
        )
        
        logger.info(f"Model downloaded to: {model_path}")
        
        # Load YOLO model
        model = YOLO(model_path)
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model = None
        raise e
    finally:
        model_loading = False

# # @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Manage application lifespan"""
#     # Startup
#     logger.info("🚀 Starting Face Mask Detection API...")
#     await load_model()
#     yield
#     # Shutdown
#     logger.info("👋 Shutting down API...")