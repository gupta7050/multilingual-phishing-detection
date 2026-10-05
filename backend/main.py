from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .prediction import predict_message
from .database import SessionLocal
from .models import PredictionHistory


app = FastAPI(
    title="Multilingual Phishing Detection API",
    version="1.0.0"
)


class MessageRequest(BaseModel):
    message: str
    user_id: int = 1


@app.get("/")
def root():
    return {
        "message": "Phishing Detection API is running"
    }


@app.post("/predict")
def predict(request: MessageRequest):

    result = predict_message(request.message)

    db: Session = SessionLocal()

    try:
        history = PredictionHistory(
            user_id=request.user_id,
            message=request.message,
            language="Unknown",
            prediction=result["prediction"],
            confidence=result["confidence"]
        )

        db.add(history)
        db.commit()
        db.refresh(history)

    finally:
        db.close()

    return {
        "message": request.message,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "history_id": history.id
    }