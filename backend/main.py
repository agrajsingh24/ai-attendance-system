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
    ForeignKey, DateTime, Float, extract
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from insightface.app import FaceAnalysis

# =========================================================
# CONFIG
# =========================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./students.db").strip()
SECRET_KEY = "CHANGE_THIS_SECRET_KEY"
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
    role = Column(String)  # student / teacher / admin


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
def register(name: str, email: str, password: str, role: str, db: Session = Depends(get_db)):
    user = User(name=name, email=email, password=hash_password(password), role=role)
    db.add(user)
    db.commit()
    return {"message": "Registered"}

@app.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(401, "Invalid credentials")
    token = create_token({"user_id": user.id})
    return {"access_token": token}

# =========================================================
# TEACHER MARK ATTENDANCE
# =========================================================

@app.post("/teacher/mark-attendance")
async def mark_attendance(
    class_name: str = Form(...),
    subject: str = Form(...),
    period: int = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_user),
    db: Session = Depends(get_db)
):
    require_role(user, ["teacher", "admin"])

    model = get_model()
    img = await file.read()
    image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    faces = model.get(image)

    session = AttendanceSession(
        teacher_id=user.id,
        class_name=class_name,
        subject=subject,
        date=str(datetime.now().date()),
        period=period,
        locked_until=datetime.utcnow() + timedelta(hours=24)
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    students = db.query(Student).all()

    for student in students:
        known = pickle.loads(bytes.fromhex(student.embedding))
        status = "absent"
        conf = 0.0

        for face in faces:
            emb = normalize(face.embedding)
            dist = np.linalg.norm(known - emb)
            if dist < MATCH_THRESHOLD:
                status = "present"
                conf = float(dist)

        record = AttendanceRecord(
            session_id=session.id,
            student_id=student.id,
            status=status,
            confidence=conf
        )
        db.add(record)

        await notify(student.user_id, {
            "date": session.date,
            "period": session.period,
            "subject": subject,
            "status": status
        })

    db.commit()
    return {"message": "Attendance marked"}

# =========================================================
# TEACHER EDIT (24 HOURS)
# =========================================================

@app.put("/teacher/edit/{record_id}")
def edit_attendance(
    record_id: int,
    new_status: str,
    user: User = Depends(get_user),
    db: Session = Depends(get_db)
):
    require_role(user, ["teacher", "admin"])

    record = db.query(AttendanceRecord).filter_by(id=record_id).first()
    session = db.query(AttendanceSession).filter_by(id=record.session_id).first()

    if user.role == "teacher" and datetime.utcnow() > session.locked_until:
        raise HTTPException(403, "Edit window expired")

    record.status = new_status
    record.edited_by = user.role
    record.edited_at = datetime.utcnow()

    db.commit()
    return {"message": "Attendance updated"}

# =========================================================
# ADMIN REPORTS
# =========================================================

@app.get("/admin/defaulters")
def defaulters(db: Session = Depends(get_db), user: User = Depends(get_user)):
    require_role(user, ["admin"])

    result = []
    students = db.query(Student).all()

    for student in students:
        total = db.query(AttendanceRecord).filter_by(student_id=student.id).count()
        present = db.query(AttendanceRecord).filter_by(
            student_id=student.id, status="present"
        ).count()

        percentage = (present / total * 100) if total > 0 else 0

        if percentage < 75:
            result.append({
                "student_id": student.id,
                "attendance_percentage": round(percentage, 2)
            })

    return {"defaulters": result}

# =========================================================
# SUBJECT ANALYTICS
# =========================================================

@app.get("/admin/subject-analytics")
def subject_analytics(subject: str, db: Session = Depends(get_db), user: User = Depends(get_user)):
    require_role(user, ["admin"])

    sessions = db.query(AttendanceSession).filter_by(subject=subject).all()

    total = 0
    present = 0

    for s in sessions:
        records = db.query(AttendanceRecord).filter_by(session_id=s.id).all()
        total += len(records)
        present += len([r for r in records if r.status == "present"])

    percentage = (present / total * 100) if total > 0 else 0

    return {
        "subject": subject,
        "attendance_percentage": round(percentage, 2)
    }

# =========================================================
# CSV EXPORT
# =========================================================

@app.get("/admin/export/{class_name}")
def export_csv(class_name: str, db: Session = Depends(get_db), user: User = Depends(get_user)):
    require_role(user, ["admin"])

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Subject", "Status", "Date"])

    sessions = db.query(AttendanceSession).filter_by(class_name=class_name).all()

    for s in sessions:
        records = db.query(AttendanceRecord).filter_by(session_id=s.id).all()
        for r in records:
            writer.writerow([r.student_id, s.subject, r.status, s.date])

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"}
    )

# =========================================================
# DISPUTE SYSTEM
# =========================================================

@app.post("/student/dispute/{record_id}")
def dispute(record_id: int, reason: str,
            user: User = Depends(get_user),
            db: Session = Depends(get_db)):

    require_role(user, ["student"])

    dispute = AttendanceDispute(
        record_id=record_id,
        student_id=user.id,
        reason=reason
    )

    db.add(dispute)
    db.commit()

    return {"message": "Dispute submitted"}

@app.put("/admin/resolve-dispute/{dispute_id}")
def resolve_dispute(dispute_id: int, status: str,
                    user: User = Depends(get_user),
                    db: Session = Depends(get_db)):

    require_role(user, ["admin"])

    dispute = db.query(AttendanceDispute).filter_by(id=dispute_id).first()
    dispute.status = status
    db.commit()

    return {"message": "Dispute resolved"}
