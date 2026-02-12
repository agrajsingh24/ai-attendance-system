from sqlalchemy import Column, Integer, String, DateTime, Float
from datetime import datetime
from app.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String)
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
