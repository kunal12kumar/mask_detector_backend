# In this we will load our model and pass the images and videos to give the result 
# a function to lad the model

from ultralytics import YOLO 
model=YOLO("best.pt")