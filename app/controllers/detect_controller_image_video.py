# In this we will define the function of processing the image and video for the detection 
import cv2
import os 
import uuid
from ultralytics import YOLO
from app.utils.file_utils import save_upload_file, delete_file
from app.services.load_trained_model import get_model  # Import the getter function
import matplotlib.pyplot as plt

async def process_image(file):
    """Process uploaded image for mask detection"""
    file_path = save_upload_file(file)   
    
    try:
        print("getting images")
        
        # Get the loaded model
        current_model = get_model()
        
        if current_model is None:
            raise Exception("Model failed to load")
        
        print(f"Model type: {type(current_model)}")
        print(f"Processing image: {file_path}")
        
        # Read and process image
        bgr_image = cv2.imread(file_path)
        if bgr_image is None:
            raise Exception(f"Could not read image from {file_path}")
            
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        
        # Run inference
        print("Running model inference...")
        result = current_model(file_path)
        
        print(f"Inference complete. Results: {result}")
        
        detections = []  # to store detected data like confidence, number of masked people, etc.
        
        # Process detection results
        if result and len(result) > 0 and result[0].boxes is not None:
            for box in result[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append({
                    "class": current_model.names[class_id],
                    "confidence": round(confidence, 3)
                })
        else:
            print("No detections found in the image")

        print(f"Final detections: {detections}")
        return {"detections": detections}
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            delete_file(file_path)

async def process_video(file):
    """Process uploaded video for mask detection"""
    video_path = save_upload_file(file)
    output_path = video_path.replace(".mp4", "_out.mp4")

    try:
        print("Processing video...")
        
        # Get the loaded model
        current_model = get_model()
        
        if current_model is None:
            raise Exception("Model failed to load")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
            
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # Run inference on frame
            results = current_model(frame)
            annotated = results[0].plot()
            out.write(annotated)

        cap.release()
        out.release()
        
        print(f"Video processing complete. Output: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Clean up input video file
        if os.path.exists(video_path):
            delete_file(video_path)