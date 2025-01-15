from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class BorrowerBase(BaseModel):
    name: str
    email: EmailStr
    phone: str

class BorrowerCreate(BorrowerBase):
    pass

class Borrower(BorrowerBase):
    id: int

    class Config:
        from_attributes = True


class BorrowRecordBase(BaseModel):
    book_id: int
    borrower_id: int

class BorrowRecordCreate(BorrowRecordBase):
    pass

class BorrowRecord(BorrowRecordBase):
    id: int
    borrow_date: datetime
    return_date: Optional[datetime] = None

    class Config:
        from_attributes = True