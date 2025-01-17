from pydantic import BaseModel
from typing import List, Optional

# Schema cho Tag
class TagBase(BaseModel):
    name: str

class TagCreate(TagBase):
    pass  # Sử dụng khi tạo Tag mới

class TagResponse(TagBase):
    id: int
    class Config:
        from_attributes = True  # Cho phép sử dụng ORM objects (SQLAlchemy)

# Schema cho Category
class CategoryBase(BaseModel):
    name: str

class CategoryCreate(CategoryBase):
     pass  # Sử dụng khi tạo Category mới

class CategoryResponse(CategoryBase):
    id: int 

    class Config:
        from_attributes = True # Cho phép sử dụng ORM objects (SQLAlchemy)


# Schema cho Product
class ProductBase(BaseModel):
    name: str
    description: Optional[str] 

class ProductCreate(ProductBase):
    category_id: Optional[int]
    tags: List[int] = []  # Danh sách ID của các Tag liên kết

class ProductResponse(ProductBase):
    id: int
    category: Optional[CategoryResponse]
    tags: List[TagResponse] = []  # Danh sách Tag trả về đầy đủ
    class Config:
        from_attributes = True