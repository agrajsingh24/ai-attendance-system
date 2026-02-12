from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.db.database import Base, engine
from app.routers import auth, teacher, student

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Attendance Management System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(teacher.router)
app.include_router(student.router)

@app.get("/")
def root():
    return {"status": "Backend Running"}
