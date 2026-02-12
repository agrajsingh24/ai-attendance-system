import os
import pickle
import numpy as np
import cv2
import csv
from io import StringIO
from datetime import datetime, timedelta

from fastapi import (
    FastAPI, UploadFile, File, Form,
    Depends, HTTPException, WebSocket,
    WebSocketDisconnect
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String,
    ForeignKey, DateTime, Float
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from insightface.app import FaceAnalysis

# =========================================================
# CONFIG
# =========================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./students.db").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_SECRET_KEY")
ALGORITHM = "HS256"
MATCH_THRESHOLD = 1.0

# =========================================================
# DATABASE
# =========================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# =========================================================
# MODELS
# =========================================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    role = Column(String)


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    embedding = Column(String)


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer)
    class_name = Column(String)
    subject = Column(String)
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
    edited_by = Column(String, nullable=True)
    edited_at = Column(DateTime, nullable=True)


class AttendanceDispute(Base):
    __tablename__ = "attendance_disputes"
    id = Column(Integer, primary_key=True)
    record_id = Column(Integer)
    student_id = Column(Integer)
    reason = Column(String)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# =========================================================
# AUTH
# =========================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(p, h):
    return pwd_context.verify(p, h)

def create_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except JWTError:
        raise HTTPException(401, "Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user

def require_role(user, roles):
    if user.role not in roles:
        raise HTTPException(403, "Access denied")

# =========================================================
# APP
# =========================================================

app = FastAPI(title="College Attendance Production System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ROOT + HEALTH CHECK (IMPORTANT FOR DEPLOYMENT)
# =========================================================

@app.get("/")
def root():
    return {
        "message": "College Attendance Backend Running",
        "status": "success"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

# =========================================================
# WEBSOCKET
# =========================================================

active_connections = {}

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    await websocket.accept()
    active_connections[user_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.pop(user_id, None)

async def notify(user_id, message):
    if user_id in active_connections:
        await active_connections[user_id].send_json(message)

# =========================================================
# INSIGHTFACE
# =========================================================

face_model = None

def get_model():
    global face_model
    if face_model is None:
        face_model = FaceAnalysis(name="buffalo_s")
        face_model.prepare(ctx_id=-1)
    return face_model

def normalize(e):
    return e / np.linalg.norm(e)

# =========================================================
# AUTH ROUTES
# =========================================================

@app.post("/register")
def register(name: str, email: str, password: str, role: str,
             db: Session = Depends(get_db)):

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(400, "Email already registered")

    user = User(
        name=name,
        email=email,
        password=hash_password(password),
        role=role
    )

    db.add(user)
    db.commit()

    return {"message": "User registered successfully"}

@app.post("/login")
def login(email: str, password: str,
          db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.password):
        raise HTTPException(401, "Invalid credentials")

    token = create_token({"user_id": user.id})

    return {"access_token": token, "token_type": "bearer"}
