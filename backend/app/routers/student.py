from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import AttendanceRecord
from app.core.security import get_current_user, require_role

router = APIRouter(prefix="/student", tags=["Student"])

@router.get("/dashboard")
def dashboard(user=Depends(get_current_user),
              db: Session = Depends(get_db)):

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
