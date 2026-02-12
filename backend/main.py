from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from deepface import DeepFace
import shutil
import os
import json
import numpy as np
from datetime import datetime
import uuid
import logging

# ================= CONFIG =================
DATABASE_URL = "sqlite:///students.db"
UPLOAD_FOLDER = "uploads"
SIMILARITY_THRESHOLD = 0.7
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB

# ================= APP INIT =================
app = FastAPI(title="AI Attendance System 🚀")

logging.basicConfig(level=logging.INFO)

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://automated-attandance.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DATABASE =================
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    roll_no = Column(String, unique=True, nullable=False)
    dob = Column(String, nullable=False)
    embeddings = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

# ================= DB DEPENDENCY =================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================= STORAGE =================
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= MIDDLEWARE (UPLOAD LIMIT) =================
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.headers.get("content-length"):
        if int(request.headers["content-length"]) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (Max 5MB)")
    return await call_next(request)

# ================= HELPER FUNCTIONS =================
def get_embedding(image_path):
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=True
        )
        return embedding[0]["embedding"]
    except Exception:
        raise HTTPException(status_code=400, detail="No face detected or invalid image")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ================= ROUTES =================

@app.get("/")
def root():
    return {"message": "DeepFace Attendance Backend Running 🚀"}

# -------- REGISTER STUDENT --------
@app.post("/register/")
async def register_student(
    name: str = Form(...),
    roll_no: str = Form(...),
    dob: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    existing = db.query(Student).filter(Student.roll_no == roll_no).first()
    if existing:
        raise HTTPException(status_code=400, detail="Roll number already registered")

    student_folder = os.path.join(UPLOAD_FOLDER, roll_no)
    os.makedirs(student_folder, exist_ok=True)

    all_embeddings = []

    for file in files:
        file_path = os.path.join(student_folder, f"{uuid.uuid4().hex}.jpg")

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

    logging.info(f"Registered student {roll_no}")

    return {"message": "Student Registered Successfully ✅"}

# -------- MARK ATTENDANCE --------
@app.post("/attendance/")
async def mark_attendance(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    temp_filename = f"classroom_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        detected_faces = DeepFace.extract_faces(
            img_path=temp_path,
            enforce_detection=True
        )

        classroom_embeddings = []

        for face in detected_faces:
            face_embedding = DeepFace.represent(
                img_path=face["face"],
                model_name="Facenet",
                enforce_detection=False
            )
            classroom_embeddings.append(face_embedding[0]["embedding"])

        if not classroom_embeddings:
            raise HTTPException(status_code=400, detail="No faces detected")

        students = db.query(Student).all()

        present_students = []
        absent_students = []

        for student in students:
            stored_embeddings = json.loads(student.embeddings)
            matched = False

            for class_emb in classroom_embeddings:
                for stored_emb in stored_embeddings:
                    if cosine_similarity(class_emb, stored_emb) > SIMILARITY_THRESHOLD:
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
            "time": datetime.now().strftime("%H:%M:%S"),
            "present": list(set(present_students)),
            "absent": absent_students
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------- GET STUDENTS --------
@app.get("/students/")
def get_students(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    return [
        {"name": s.name, "roll_no": s.roll_no, "dob": s.dob}
        for s in students
    ]
