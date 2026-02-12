import os
import pickle
import numpy as np
import cv2
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import (
    FastAPI, Depends, HTTPException,
    UploadFile, File, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String,
    ForeignKey, DateTime, Float, Boolean,
    Index
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from insightface.app import FaceAnalysis

# =========================================================
# CONFIG
# =========================================================

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "SUPER_SECRET_CHANGE_THIS")
ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
MATCH_THRESHOLD = 1.0

# =========================================================
# DATABASE
# =========================================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# =========================================================
# MODELS (Optimized + Indexed)
# =========================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String, index=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    token = Column(String, unique=True)
    expires_at = Column(DateTime)


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, index=True)
    class_name = Column(String, index=True)
    subject = Column(String, index=True)
    date = Column(String, index=True)
    period = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_class_subject_date", "class_name", "subject", "date"),
    )


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, index=True)
    student_id = Column(Integer, index=True)
    status = Column(String, index=True)
    confidence = Column(Float)
    marked_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_student_session", "student_id", "session_id"),
    )


Base.metadata.create_all(bind=engine)

# =========================================================
# AUTH
# =========================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(p):
    return pwd_context.hash(p)

def verify_password(p, h):
    return pwd_context.verify(p, h)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
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
# SCHEMAS
# =========================================================

class RegisterSchema(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str

class LoginSchema(BaseModel):
    email: EmailStr
    password: str

class RefreshSchema(BaseModel):
    refresh_token: str

# =========================================================
# APP
# =========================================================

app = FastAPI(title="Enterprise Attendance System (2K Ready)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Production Running"}

# =========================================================
# AUTH ROUTES
# =========================================================

@app.post("/register")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Email already exists")

    user = User(
        name=data.name,
        email=data.email,
        password=hash_password(data.password),
        role=data.role,
        is_verified=True  # set False when email system added
    )

    db.add(user)
    db.commit()
    return {"message": "User created"}

@app.post("/login")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(401, "Invalid credentials")

    access = create_access_token({"user_id": user.id})
    refresh = create_refresh_token({"user_id": user.id})

    db.add(RefreshToken(
        user_id=user.id,
        token=refresh,
        expires_at=datetime.utcnow() + timedelta(days=7)
    ))
    db.commit()

    return {
        "access_token": access,
        "refresh_token": refresh,
        "role": user.role
    }

@app.post("/refresh")
def refresh_token(data: RefreshSchema,
                  db: Session = Depends(get_db)):

    token_entry = db.query(RefreshToken).filter(
        RefreshToken.token == data.refresh_token
    ).first()

    if not token_entry or token_entry.expires_at < datetime.utcnow():
        raise HTTPException(401, "Invalid refresh token")

    access = create_access_token({"user_id": token_entry.user_id})
    return {"access_token": access}

# =========================================================
# ROLE-BASED DASHBOARD APIs
# =========================================================

@app.get("/dashboard/student")
def student_dashboard(user=Depends(get_current_user),
                      db: Session = Depends(get_db)):

    require_role(user, ["student"])

    total = db.query(AttendanceRecord).filter_by(
        student_id=user.id).count()

    present = db.query(AttendanceRecord).filter_by(
        student_id=user.id,
        status="present"
    ).count()

    percentage = (present / total * 100) if total else 0

    return {
        "total_classes": total,
        "attendance_percentage": round(percentage, 2)
    }

@app.get("/dashboard/admin")
def admin_dashboard(page: int = 1,
                    limit: int = 20,
                    user=Depends(get_current_user),
                    db: Session = Depends(get_db)):

    require_role(user, ["admin"])

    offset = (page - 1) * limit

    students = db.query(User).filter(
        User.role == "student"
    ).offset(offset).limit(limit).all()

    return {
        "page": page,
        "students": [
            {"id": s.id, "name": s.name, "email": s.email}
            for s in students
        ]
    }
