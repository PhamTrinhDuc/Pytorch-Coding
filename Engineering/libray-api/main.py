from fastapi import FastAPI, APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import BookSchemas, BookCreate
from models import Book
from database import get_db, engine


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/books", response_model=BookSchemas)
def create_book(book: BookCreate, 
                db: Session = Depends(get_db)):
    print(f"Accessing /books/ endpoint")
    db_book = Book(**book.dict())
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    return db_book


@app.get("/books", response_model=list[BookSchemas])
def read_books(skip: int = 0, limit: int = 100, 
               db: Session = Depends(get_db)):
    books = db.query(Book).offset(skip).limit(limit)
    return books

