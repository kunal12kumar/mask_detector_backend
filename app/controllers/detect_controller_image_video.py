#in this  we will define the function of proccessing the image and video for the detection 
import cv2
import os 
import uuid
from ultralytics import yolo
from app.utils.file_utils import save_upload_file , delete_file
from app.services.load_trained_model import model
import matplotlib.pyplot as plt



async def detect_image(file):
    file_path=save_upload_file(file)   # here we are fetching the image path so, we can work on that
    
    bgr_image=cv2.imread(file_path)
    rgb_image=cv2.cvtColor(bgr_image , cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    
    result=model(file_path)
    
    print(result)
    
    detections=[]  #to store data the which we detect like confidenc no.  of masked peope and other 
    
    for box in result[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        detections.append({
            "class": model.names[class_id],
            "confidence": round(confidence, 3)
        })

    delete_file(file_path)
    return {"detections": detections}


async def process_video(file):
    video_path = save_upload_file(file)
    output_path = video_path.replace(".mp4", "_out.mp4")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    delete_file(video_path)

    return output_path
    
    