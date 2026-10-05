# ============================================================
# MULTILINGUAL PHISHING DETECTION - FASTAPI BACKEND
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from sqlalchemy.exc import IntegrityError

from .prediction import predict_message
from .database import SessionLocal
from .models import User, PredictionHistory

import hashlib


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Multilingual Phishing Detection API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PASSWORD FUNCTIONS
# ============================================================

def hash_password(password: str) -> str:
    """
    Hash password using SHA-256.
    """
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def verify_password(
    password: str,
    stored_password: str
) -> bool:

    hashed_password = hash_password(password)

    # Normal hashed password
    if stored_password == hashed_password:
        return True

    # Allows the existing temporary test user
    # to continue working if its password is stored
    # as plain text.
    if stored_password == password:
        return True

    return False


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):

    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):

    email: EmailStr
    password: str


class MessageRequest(BaseModel):

    message: str
    user_id: int


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Phishing Detection API is running"
    }


# ============================================================
# REGISTER
# ============================================================

@app.post("/register")
def register(request: RegisterRequest):

    db: Session = SessionLocal()

    try:

        # Check if email already exists
        existing_user = (
            db.query(User)
            .filter(User.email == request.email)
            .first()
        )

        if existing_user:

            raise HTTPException(
                status_code=400,
                detail="Email already registered."
            )

        # Create user
        new_user = User(
            name=request.name,
            email=request.email,
            password_hash=hash_password(
                request.password
            )
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "message": "Registration successful",
            "user_id": new_user.id,
            "name": new_user.name,
            "email": new_user.email
        }

    except HTTPException:

        raise

    except IntegrityError:

        db.rollback()

        raise HTTPException(
            status_code=400,
            detail="Email already registered."
        )

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )

    finally:

        db.close()


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
def login(request: LoginRequest):

    db: Session = SessionLocal()

    try:

        user = (
            db.query(User)
            .filter(User.email == request.email)
            .first()
        )

        if not user:

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        if not verify_password(
            request.password,
            user.password_hash
        ):

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        # If old plain-text test password was used,
        # automatically convert it to a hash.
        if user.password_hash == request.password:

            user.password_hash = hash_password(
                request.password
            )

            db.commit()

        return {
            "message": "Login successful",
            "user_id": user.id,
            "name": user.name,
            "email": user.email
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )

    finally:

        db.close()


# ============================================================
# PREDICT MESSAGE
# ============================================================

@app.post("/predict")
def predict(request: MessageRequest):

    if not request.message.strip():

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    db: Session = SessionLocal()

    try:

        # Check user exists
        user = (
            db.query(User)
            .filter(User.id == request.user_id)
            .first()
        )

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User not found."
            )

        # Run ML prediction
        result = predict_message(
            request.message
        )

        # Save prediction history
        history = PredictionHistory(
            user_id=request.user_id,
            message=request.message,
            language=result.get(
                "language",
                "Unknown"
            ),
            prediction=result["prediction"],
            confidence=result["confidence"]
        )

        db.add(history)
        db.commit()
        db.refresh(history)

        return {
            "message": request.message,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "language": result.get(
                "language",
                "Unknown"
            ),
            "history_id": history.id
        }

    except HTTPException:

        raise

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:

        db.close()


# ============================================================
# GET USER HISTORY
# ============================================================

@app.get("/history/{user_id}")
def get_history(user_id: int):

    db: Session = SessionLocal()

    try:

        # Check user exists
        user = (
            db.query(User)
            .filter(User.id == user_id)
            .first()
        )

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User not found."
            )

        history = (
            db.query(PredictionHistory)
            .filter(
                PredictionHistory.user_id == user_id
            )
            .order_by(
                PredictionHistory.created_at.desc()
            )
            .all()
        )

        return [
            {
                "id": item.id,
                "message": item.message,
                "language": item.language,
                "prediction": item.prediction,
                "confidence": item.confidence,
                "created_at": item.created_at
            }
            for item in history
        ]

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch history: {str(e)}"
        )

    finally:

        db.close()
