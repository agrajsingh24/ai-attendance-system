import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine, Column, Integer, String,
    ForeignKey, DateTime, Float, Boolean
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# =========================================================
# CONFIG
# =========================================================

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(title="Attendance Management System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DATABASE MODELS
# =========================================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String)  # student / teacher / admin
    created_at = Column(DateTime, default=datetime.utcnow)


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, index=True)
    class_name = Column(String)
    subject = Column(String)
    period = Column(Integer)
    date = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, index=True)
    student_id = Column(Integer, index=True)
    status = Column(String)
    confidence = Column(Float)
    marked_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# =========================================================
# UTILITIES
# =========================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_role(user, roles):
    if user.role not in roles:
        raise HTTPException(status_code=403, detail="Access denied")


# =========================================================
# SCHEMAS
# =========================================================

class RegisterSchema(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str  # student / teacher / admin


class LoginSchema(BaseModel):
    email: EmailStr
    password: str


class CreateSessionSchema(BaseModel):
    class_name: str
    subject: str
    period: int


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {"message": "Attendance System Running"}

# =========================================================
# AUTH
# =========================================================

@app.post("/register")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = User(
        name=data.name,
        email=data.email,
        password=hash_password(data.password),
        role=data.role
    )
    db.add(user)
    db.commit()

    return {"message": "User registered successfully"}


@app.post("/login")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"user_id": user.id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role
    }

# =========================================================
# TEACHER ROUTES
# =========================================================

@app.post("/teacher/create-session")
def create_session(
    data: CreateSessionSchema,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    require_role(user, ["teacher"])

    session = AttendanceSession(
        teacher_id=user.id,
        class_name=data.class_name,
        subject=data.subject,
        period=data.period,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "message": "Session created",
        "session_id": session.id
    }


@app.post("/teacher/upload-class-photo/{session_id}")
async def upload_class_photo(
    session_id: int,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    require_role(user, ["teacher"])

    session = db.query(AttendanceSession).filter(
        AttendanceSession.id == session_id,
        AttendanceSession.teacher_id == user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    contents = await file.read()

    file_path = f"class_photo_{session_id}.jpg"
    with open(file_path, "wb") as f:
        f.write(contents)

    # DEMO LOGIC (marks all students present)
    students = db.query(User).filter(User.role == "student").all()

    for student in students:
        record = AttendanceRecord(
            session_id=session_id,
            student_id=student.id,
            status="present",
            confidence=0.95
        )
        db.add(record)

    db.commit()

    return {"message": "Attendance marked successfully"}

# =========================================================
# STUDENT DASHBOARD
# =========================================================

@app.get("/dashboard/student")
def student_dashboard(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    require_role(user, ["student"])

    total = db.query(AttendanceRecord).filter(
        AttendanceRecord.student_id == user.id
    ).count()

    present = db.query(AttendanceRecord).filter(
        AttendanceRecord.student_id == user.id,
        AttendanceRecord.status == "present"
    ).count()

    percentage = (present / total * 100) if total else 0

    return {
        "total_classes": total,
        "attendance_percentage": round(percentage, 2)
    }

# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.get("/dashboard/admin")
def admin_dashboard(
    page: int = 1,
    limit: int = 20,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
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

