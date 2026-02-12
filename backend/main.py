from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import numpy as np
import os
import shutil

app = FastAPI()

# Allow frontend (Netlify) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "AI Attendance System Backend Running"}


@app.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run DeepFace verification
        result = DeepFace.verify(
            img1_path=file_path,
            img2_path=file_path,  # Replace with stored image path in real use
            model_name="VGG-Face",
            enforce_detection=False
        )

        return {
            "verified": result["verified"],
            "distance": result["distance"]
        }

    except Exception as e:
        return {"error": str(e)}
