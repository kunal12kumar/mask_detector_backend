# In this we define everything about fast api actually initialize the fast api here for or it is the entry point of the fastapi 

from fastapi import FastAPI
from app.routers import detect

app=FastAPI(
     title="Mask Detection API",
     version="1.0"
)

# Register routers
app.include_router(detect.router)