from pydantic import BaseModel
from typing import List, Optional
from .book import Book

class AuthorBase(BaseModel):
    name: str
    bio: Optional[str] = None

class AuthorCreate(AuthorBase):
    pass

class Author(AuthorBase):
    id: int
    books: List[Book] = []

    class Config:
        orm_mode = True