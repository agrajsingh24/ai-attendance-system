from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.database import get_db
from app.db.models import AttendanceSession, AttendanceRecord, User
from app.schemas.attendance import CreateSessionSchema
from app.core.security import get_current_user, require_role

router = APIRouter(prefix="/teacher", tags=["Teacher"])

@router.post("/create-session")
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

    return {"session_id": session.id}

@router.post("/upload-photo/{session_id}")
async def upload_photo(
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

    return {"message": "Attendance marked"}
