import os
import pickle
import numpy as np
import cv2

from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String,
    ForeignKey, DateTime, Float
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./students.db").strip()

SECRET_KEY = "supersecretkey"  # change later
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

MATCH_THRESHOLD = 1.0

# =========================
# DATABASE
# =========================

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# =========================
# MODELS
# =========================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    role = Column(String)  # student / teacher


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    roll_no = Column(String, unique=True)
    embedding = Column(String)  # stored as pickle string


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer)
    class_name = Column(String)
    date = Column(String)
    period = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    locked_until = Column(DateTime)


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    student_id = Column(Integer)
    status = Column(String)
    confidence = Column(Float)
    marked_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# =========================
# AUTH
# =========================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# =========================
# FASTAPI INIT
# =========================

app = FastAPI(title="College Attendance System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# INSIGHTFACE (Lazy Load)
# =========================

face_model = None

def get_model():
    global face_model
    if face_model is None:
        face_model = FaceAnalysis(name="buffalo_s")
        face_model.prepare(ctx_id=-1)
    return face_model

def normalize(e):
    return e / np.linalg.norm(e)

# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {"status": "Backend Running"}

# -------------------------
# REGISTER
# -------------------------

@app.post("/register")
def register(name: str, email: str, password: str, role: str,
             db: Session = Depends(get_db)):

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already exists")

    user = User(
        name=name,
        email=email,
        password=hash_password(password),
        role=role
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "User registered"}

# -------------------------
# LOGIN
# -------------------------

@app.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.password):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({"user_id": user.id})

    return {"access_token": token}

# -------------------------
# REGISTER FACE (Student)
# -------------------------

@app.post("/student/register-face")
async def register_face(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "student":
        raise HTTPException(403, "Only students allowed")

    model = get_model()

    img = await file.read()
    np_img = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = model.get(image)

    if len(faces) != 1:
        raise HTTPException(400, "One face required")

    embedding = normalize(faces[0].embedding)

    student = Student(
        user_id=current_user.id,
        roll_no=str(current_user.id),
        embedding=pickle.dumps(embedding).hex()
    )

    db.add(student)
    db.commit()

    return {"message": "Face registered"}

# -------------------------
# TEACHER MARK ATTENDANCE
# -------------------------

@app.post("/teacher/mark-attendance")
async def mark_attendance(
    class_name: str = Form(...),
    period: int = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "teacher":
        raise HTTPException(403, "Only teachers allowed")

    model = get_model()

    img = await file.read()
    np_img = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = model.get(image)

    students = db.query(Student).all()

    session = AttendanceSession(
        teacher_id=current_user.id,
        class_name=class_name,
        date=str(datetime.now().date()),
        period=period,
        locked_until=datetime.utcnow() + timedelta(hours=24)
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    for student in students:
        known = pickle.loads(bytes.fromhex(student.embedding))
        status = "absent"
        confidence = 0.0

        for face in faces:
            emb = normalize(face.embedding)
            dist = np.linalg.norm(known - emb)
            if dist < MATCH_THRESHOLD:
                status = "present"
                confidence = float(dist)

        record = AttendanceRecord(
            session_id=session.id,
            student_id=student.id,
            status=status,
            confidence=confidence
        )
        db.add(record)

    db.commit()

    return {"message": "Attendance marked"}


# =========================
# RAILWAY PORT BINDING FIX
# =========================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
