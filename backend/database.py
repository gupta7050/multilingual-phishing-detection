import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import URL

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("MYSQLUSER", "root"),
    password=os.getenv("MYSQLPASSWORD", ""),
    host=os.getenv("MYSQLHOST", "localhost"),
    port=int(os.getenv("MYSQLPORT", "3306")),
    database=os.getenv("MYSQLDATABASE", "phishing_detection")
)

engine = create_engine(
    DATABASE_URL,
    echo=False
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
