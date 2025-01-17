import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from model import Base, Category, Product, Tag

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(url=DATABASE_URL)
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(bind=engine)

# Dependency để lấy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_data():
    db = SessionLocal()

    # Thêm danh mục
    category = Category(name="Electronics")
    db.add(category)

    # Thêm sản phẩm
    product1 = Product(name="Smartphone", description="Điện thoại thông minh", category=category)
    # category=category: ánh xạ sản phẩm này vào danh mục
    product2 = Product(name="Laptop", description="Máy tính xách tay", category=category)
    db.add_all([product1, product2])

    # Thêm thẻ
    tag1 = Tag(name="Portable")
    tag2 = Tag(name="Expensive")
    # Thêm cả hai thẻ tag1 và tag2 vào danh sách thẻ của sản phẩm product1.
    product1.tags.extend([tag1, tag2])
    product2.tags.extend([tag2])

    db.commit()
    print("Thêm dữ liệu thành công!")


def query():
    db = SessionLocal()

    # Lấy tất cả sản phẩm trong một danh mục
    category = db.query(Category).filter_by(name="Electronics").first()
    if category:
        print(f"Danh sách sản phẩm trong danh mục {category.name}:")
        for product in category.products:
            print(product)

    # Lấy tất cả thẻ của một sản phẩm
    product = db.query(Product).filter_by(name="Smartphone").first()
    if product:
        print(f"Thẻ của sản phẩm {product.name}:")
        for tag in product.tags:
            print(tag)

    # Lấy tất cả sản phẩm có một thẻ cụ thể
    tag = db.query(Tag).filter_by(name="Expensive").first()
    if tag:
        print(f"Sản phẩm có thẻ {tag.name}:")
    for product in tag.products:
        print(product)

    # Cập nhật danh mục cho một sản phẩm
    product = db.query(Product).filter_by(name="Laptop").first()
    new_category = Category(name="Computers")
    if product:
        db.add(new_category)
        product.category = new_category
        db.commit()
        print(f"Đã cập nhật danh mục của {product.name} thành {new_category.name}")

    # Xóa dữ liệu
    tag = db.query(Tag).filter_by(name="Portable").first()
    if tag:
        db.delete(tag)
        db.commit()
        print(f"Đã xóa thẻ: {tag}")


def main():
    from sqlalchemy import inspect
    add_data()
    query()

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(tables)  # Danh sách các bảng trong cơ sở dữ liệu

if __name__ == "__main__":
    main()