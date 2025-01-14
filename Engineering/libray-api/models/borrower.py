from sqlalchemy import (
    Column, 
    String, 
    Integer, 
    ForeignKey, 
    DateTime
)
from sqlalchemy.orm import relationship
from ..database import Base
from datetime import datetime


class Borrower(Base):

    __tablename__ = "borrowers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    phone = Column(String)

    borrowed_books = relationship("BorrowRecord", back_populates="borrower")

class BorrowRecord(Base):

    __tablename__ = "borrow_records"

    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(Integer, ForeignKey("books.id"))
    borrower_id = Column(Integer, "borrowers.id")
    borrow_date = Column(DateTime, default=datetime.utcnow)
    return_date = Column(DateTime, nullable=True)

    book = relationship("Book")
    borrower = relationship("Borrower", back_populates="borrowed_books")
