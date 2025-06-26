# the code to define routers 
from fastapi import APIRouter , UploadFile , File
from fastapi.responses import JSONResponse , FileResponse 
from app.controllers.detect_controller_image_video import process_video , process_image


router=APIRouter(prefix="/detect" ,tags=["Detection"])

@router.post('/image')
async def detect__mask_of_image(file: UploadFile = File(...)):
     result = await process_image(file)
     print(result)
     return JSONResponse(content=result)
 
 
# now for the video

@router.post('/video')
async def detect_mask_of_video(file: UploadFile = File(...)):
    output_path=await process_video(file)

    return FileResponse(output_path, media_type="video/mp4", filename="output_output.mp4")

