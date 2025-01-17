from sqlalchemy import (
    Column, Integer, String, ForeignKey, Table
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# Bảng trung gian cho quan hệ nhiều-nhiều giữa Product và Tag
product_tag = Table(
    'product_tag', Base.metadata,
    Column('product_id', Integer, ForeignKey('products.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

# Bảng Category (1-nhiều với Product)
class Category(Base):
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    products = relationship("Product", back_populates="category")

    def __repr__(self):
        return f"<Category(name={self.name})>"

# Bảng Product (nhiều-1 với Category, nhiều-nhiều với Tag)
class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String)
    category_id = Column(Integer, ForeignKey('categories.id'))
    category = relationship("Category", back_populates="products")
    tags = relationship(
        "Tag", secondary=product_tag, back_populates="products"
    )

    def __repr__(self):
        return f"<Product(name={self.name}, category={self.category.name})>"

# Bảng Tag (nhiều-nhiều với Product)
class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    products = relationship(
        "Product", secondary=product_tag, back_populates="tags"
    )

    def __repr__(self):
        return f"<Tag(name={self.name})>"