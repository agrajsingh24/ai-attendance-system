from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base
import face_recognition
import numpy as np
import os
import shutil
import uuid
import pickle
import cv2
from datetime import datetime

# =========================
# CONFIG
# =========================

DATABASE_URL = "sqlite:///./students.db"
UPLOAD_FOLDER = "temp_uploads"
MATCH_THRESHOLD = 0.5  # 0.4 strict | 0.5 balanced | 0.6 lenient

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
    embedding = Column(LargeBinary, nullable=False)  # store 128d encoding


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

app = FastAPI(title="AI Attendance System (face-recognition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {"message": "AI Attendance Backend Running (face-recognition version)"}


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

    encodings = []

    for file in files:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_encodings(rgb)

        if len(faces) != 1:
            raise HTTPException(
                status_code=400,
                detail="Each selfie must contain exactly one clear face"
            )

        encodings.append(faces[0])

    # Average embedding
    avg_encoding = np.mean(encodings, axis=0)

    embedding_bytes = pickle.dumps(avg_encoding)

    new_student = Student(
        name=name,
        roll_no=roll_no,
        dob=dob,
        embedding=embedding_bytes
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
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect multiple faces
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if len(face_encodings) == 0:
        raise HTTPException(status_code=400, detail="No faces detected")

    students = db.query(Student).all()

    if not students:
        raise HTTPException(status_code=400, detail="No registered students")

    known_encodings = []
    known_names = []

    for student in students:
        encoding = pickle.loads(student.embedding)
        known_encodings.append(encoding)
        known_names.append(student.name)

    present_students = set()

    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(distances) == 0:
            continue

        min_distance = np.min(distances)
        best_match_index = np.argmin(distances)

        if min_distance < MATCH_THRESHOLD:
            name = known_names[best_match_index]
            present_students.add(name)

    all_students = set(known_names)
    absent_students = list(all_students - present_students)

    return {
        "date": str(datetime.now().date()),
        "time": datetime.now().strftime("%H:%M:%S"),
        "present": list(present_students),
        "absent": absent_students
    }
