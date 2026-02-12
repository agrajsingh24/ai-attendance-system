from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import pickle
from datetime import datetime

# =========================
# CONFIG
# =========================

DATABASE_URL = "sqlite:///./students.db"

# 0.6 is strict
# 0.65 slightly relaxed (better for group photos)
MATCH_THRESHOLD = 0.65  

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
    embedding = Column(LargeBinary, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# FASTAPI INIT
# =========================

app = FastAPI(title="AI Attendance System (InsightFace - Railway Optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD INSIGHTFACE MODEL
# =========================

app_face = None

@app.on_event("startup")
def load_model():
    global app_face
    app_face = FaceAnalysis(name="buffalo_s")  # lightweight model
    app_face.prepare(ctx_id=-1)  # CPU only

# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {"message": "InsightFace Attendance Backend Running"}

# =========================
# REGISTER STUDENT
# =========================

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
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        faces = app_face.get(image)

        if len(faces) != 1:
            raise HTTPException(
                status_code=400,
                detail="Each selfie must contain exactly one face"
            )

        embeddings.append(faces[0].embedding.astype(np.float32))

    # Average embedding (master embedding)
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

    embedding_bytes = pickle.dumps(avg_embedding)

    new_student = Student(
        name=name,
        roll_no=roll_no,
        dob=dob,
        embedding=embedding_bytes
    )

    db.add(new_student)
    db.commit()

    return {"message": "Student registered successfully"}

# =========================
# MARK ATTENDANCE
# =========================

@app.post("/attendance")
async def mark_attendance(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = app_face.get(image)

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected")

    students = db.query(Student).all()

    if not students:
        raise HTTPException(status_code=400, detail="No registered students")

    # Load known embeddings
    known_encodings = []
    known_names = []

    for student in students:
        encoding = pickle.loads(student.embedding).astype(np.float32)
        known_encodings.append(encoding)
        known_names.append(student.name)

    known_encodings = np.array(known_encodings)

    present_students = set()

    # For each detected face
    for face in faces:
        face_embedding = face.embedding.astype(np.float32)

        # Compute distances to ALL students
        distances = np.linalg.norm(
            known_encodings - face_embedding,
            axis=1
        )

        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]

        # Debug (optional)
        # print("Min distance:", min_distance)

        if min_distance < MATCH_THRESHOLD:
            present_students.add(known_names[best_match_index])

    all_students = set(known_names)
    absent_students = list(all_students - present_students)

    return {
        "date": str(datetime.now().date()),
        "time": datetime.now().strftime("%H:%M:%S"),
        "present": list(present_students),
        "absent": absent_students
    }
