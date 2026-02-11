from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from deepface import DeepFace
import shutil
import os
import json
import numpy as np
from datetime import datetime

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "sqlite:///./students.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ================= DATABASE MODELS =================

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    roll_no = Column(String, unique=True)
    dob = Column(String)
    embeddings = Column(Text)  # Stored as JSON

Base.metadata.create_all(bind=engine)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= HELPER FUNCTIONS =================

def get_embedding(image_path):
    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )
    return embedding[0]["embedding"]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ================= ROUTES =================

@app.get("/")
def root():
    return {"message": "DeepFace Attendance Backend Running 🚀"}

# -------- STUDENT REGISTRATION --------

@app.post("/register/")
async def register_student(
    name: str = Form(...),
    roll_no: str = Form(...),
    dob: str = Form(...),
    files: list[UploadFile] = File(...)
):
    db = SessionLocal()

    student_folder = os.path.join(UPLOAD_FOLDER, roll_no)
    os.makedirs(student_folder, exist_ok=True)

    all_embeddings = []

    for file in files:
        file_path = os.path.join(student_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        embedding = get_embedding(file_path)
        all_embeddings.append(embedding)

    student = Student(
        name=name,
        roll_no=roll_no,
        dob=dob,
        embeddings=json.dumps(all_embeddings)
    )

    db.add(student)
    db.commit()

    return {"message": "Student Registered Successfully ✅"}

# -------- ATTENDANCE --------

@app.post("/attendance/")
async def mark_attendance(file: UploadFile = File(...)):
    db = SessionLocal()

    temp_path = os.path.join(UPLOAD_FOLDER, "classroom.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detected_faces = DeepFace.extract_faces(
        img_path=temp_path,
        enforce_detection=False
    )

    classroom_embeddings = []

    for face in detected_faces:
        face_embedding = DeepFace.represent(
            img_path=face["face"],
            model_name="Facenet",
            enforce_detection=False
        )
        classroom_embeddings.append(face_embedding[0]["embedding"])

    students = db.query(Student).all()

    present_students = []
    absent_students = []

    for student in students:
        stored_embeddings = json.loads(student.embeddings)
        matched = False

        for class_emb in classroom_embeddings:
            for stored_emb in stored_embeddings:
                similarity = cosine_similarity(class_emb, stored_emb)
                if similarity > 0.7:  # Threshold
                    matched = True
                    break
            if matched:
                break

        if matched:
            present_students.append(student.name)
        else:
            absent_students.append(student.name)

    return {
        "date": str(datetime.now().date()),
        "present": present_students,
        "absent": absent_students
    }

# -------- GET ALL STUDENTS --------

@app.get("/students/")
def get_students():
    db = SessionLocal()
    return db.query(Student).all()