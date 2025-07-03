# In this we will load our model and pass the images and videos to give the result 
# a function to load the model
import logging
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from fastapi import FastAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
model_loading = False

# Replace with your Hugging Face model repository
HF_REPO_ID = "kunal12kumardev/Face_mask_detection"  # 🔄 CHANGE THIS
MODEL_FILENAME = "best.pt"

def load_model():
    """Load model from Hugging Face Hub"""
    global model, model_loading
    
    # If model is already loaded, return it
    if model is not None:
        logger.info("Model already loaded, returning existing model")
        return model
        
    # If model is currently being loaded by another thread, wait
    if model_loading:
        logger.info("Model is currently being loaded, waiting...")
        # You might want to add a small delay here if needed
        return None
        
    model_loading = True
    logger.info("Starting model download from Hugging Face...")
    
    try:
        # Download model with cache
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir="./model_cache",
            force_download=False
        )
        
        logger.info(f"Model downloaded to: {model_path}")
        
        # Load YOLO model
        model = YOLO(model_path)
        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model classes: {model.names}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        raise e
    finally:
        model_loading = False

def get_model():
    """Get the loaded model, load it if not already loaded"""
    global model
    if model is None:
        logger.info("Model not loaded, loading now...")
        load_model()
    return model

# Load model immediately when module is imported
