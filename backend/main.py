from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text
from deepface import DeepFace
import numpy as np
import os
import shutil
import uuid
import json
from datetime import datetime

# =========================
# CONFIG
# =========================

DATABASE_URL = "sqlite:///./students.db"
UPLOAD_FOLDER = "temp_uploads"
SIMILARITY_THRESHOLD = 0.6

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# DATABASE SETUP
# =========================

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    roll_no = Column(String, unique=True, index=True)
    dob = Column(String)
    embedding = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# APP INIT
# =========================

app = FastAPI(title="AI Attendance System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# FACE UTILITIES
# =========================

def get_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        enforce_detection=True
    )
    return result[0]["embedding"]

def average_embeddings(embeddings):
    return np.mean(embeddings, axis=0).tolist()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {"message": "AI Attendance Backend Running"}

# -------------------------
# STUDENT REGISTRATION
# -------------------------

@app.post("/register")
async def register_student(
    name: str = Form(...),
    roll_no: str = Form(...),
    dob: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="Upload at least 3 selfies")

    existing = db.query(Student).filter(Student.roll_no == roll_no).first()
    if existing:
        raise HTTPException(status_code=400, detail="Roll number already exists")

    embeddings = []

    for file in files:
        temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            emb = get_embedding(temp_path)
            embeddings.append(emb)
        except:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Face not detected properly")

        os.remove(temp_path)

    avg_embedding = average_embeddings(embeddings)

    new_student = Student(
        name=name,
        roll_no=roll_no,
        dob=dob,
        embedding=json.dumps(avg_embedding)
    )

    db.add(new_student)
    db.commit()

    return {"message": "Student registered successfully"}

# -------------------------
# ATTENDANCE (MULTI FACE)
# -------------------------

@app.post("/attendance")
async def mark_attendance(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Detect faces
        faces = DeepFace.extract_faces(
            img_path=temp_path,
            enforce_detection=True
        )

        if not faces:
            raise HTTPException(status_code=400, detail="No faces detected")

        students = db.query(Student).all()

        present = set()
        absent = []

        for face in faces:
            face_embedding = DeepFace.represent(
                img_path=face["face"],
                model_name="Facenet512",
                enforce_detection=False
            )[0]["embedding"]

            for student in students:
                stored_embedding = json.loads(student.embedding)

                similarity = cosine_similarity(face_embedding, stored_embedding)

                if similarity > SIMILARITY_THRESHOLD:
                    present.add(student.name)

        for student in students:
            if student.name not in present:
                absent.append(student.name)

        return {
            "date": str(datetime.now().date()),
            "time": datetime.now().strftime("%H:%M:%S"),
            "present": list(present),
            "absent": absent
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

