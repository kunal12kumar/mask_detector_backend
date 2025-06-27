# the code to define routers 
from fastapi import APIRouter , UploadFile , File
from fastapi.responses import JSONResponse , FileResponse 
from app.controllers.detect_controller_image_video import process_video , process_image

HF_REPO_ID = "kunal12kumardev/Face_mask_detection" 

router=APIRouter()

@router.post('/detect/image')
async def detect__mask_of_image(file: UploadFile = File(...)):
     print("getting images")
     result = await process_image(file)
     print(result)
     return JSONResponse(content={"message":"Hello" ,"result":result})
 
 
# now for the video

@router.post('/detect/video')
async def detect_mask_of_video(file: UploadFile = File(...)):
    output_path=await process_video(file)

    return FileResponse(output_path, media_type="video/mp4", filename="output_output.mp4")

