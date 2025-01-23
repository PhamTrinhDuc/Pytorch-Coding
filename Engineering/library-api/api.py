from fastapi import FastAPI, APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from model import Tag, Product, Category
from database import get_db
from schemas import(
    CategoryResponse,
    CategoryCreate,
    ProductResponse, 
    ProductCreate, 
    TagResponse, 
    TagCreate
)

app = FastAPI()

@app.post("/categoty/", response_model=CategoryResponse)
def create_category(category: CategoryCreate, 
                    db: Session = Depends(get_db)):
    db_category = Category(**category.dict())
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category


@app.post("/products/", response_model=ProductResponse)
def create_book(product: ProductCreate, 
                db: Session = Depends(get_db)):
    db_book = Product(**product.dict())
    db.add(db_book)
    db.commit()
    db.flush()

    # Add tags
    if product.tags:
        # Get tag objects from database
        tags = db.query(Tag).filter(Tag.id.in_(product.tags)).all()
        db_book.tags = tags

    db.refresh(db_book)
    return db_book


@app.post("/tags/", response_model=TagResponse)
def create_tag(tag: TagCreate, db: Session = Depends(get_db)):
    db_tag = Tag(name=tag.name)
    db.add(db_tag)
    db.commit()
    db.refresh(db_tag)
    return db_tag