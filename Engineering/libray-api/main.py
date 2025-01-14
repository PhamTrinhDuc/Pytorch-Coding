from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models, schemas
from .database import get_db, engine


app = APIRouter()

@app.post("/books/", response_model=schemas.Book)
def create_book(book: schemas.BookCreate, 
                db: Session = Depends(get_db)):
    db_book = models.Book(**book.dict())
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    return db_book

@app.get("/books/", response_model=list[schemas.Book])
def read_books(skip: int = 0, limit: int = 100, 
               db: Session = Depends(get_db)):
    books = db.query(models.Book).offset(skip).limit(limit)
    return books

