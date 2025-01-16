import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

"""1. KẾT NỐI CƠ SỞ DỮ LIỆU"""

# Thay đổi thông tin kết nối phù hợp với CSDL của bạn
DATABASE_URL = os.getenv("DATABASE_URL")

# Tạo engine kết nối
engine = create_engine(DATABASE_URL)

# Kiểm tra kết nối
if engine:
    print("1. Kết nối cơ sở dữ liệu thành công!")


"""2. ĐỊNH NGHĨA BẢNG VÀ MÔ HINH DỮ LIỆU"""

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String)

    def __repr__(self):
        return f"<Product(name={self.name}, description={self.description})>"

"""3. TAO BẢNG TRONG CƠ SỞ DỮ LIỆU"""
# Tạo bảng trong CSDL
Base.metadata.create_all(engine)
print("3. Tạo bảng thành công!")


"""4. LÀM VIỆC VỚI PHIÊN <SESSION>"""
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()

"""5. THÊM DỮ LIỆU VÀO BẢNG"""
# Tạo một sản phẩm mới
new_product = Product(name="Laptop", description="Máy tính xách tay hiệu suất cao")

# Thêm vào session và commit
session.add(new_product)
session.commit()
print(f"5. Đã thêm sản phẩm: {new_product}")


"""6. TRUY VẤN DỮ LIỆU"""
# Lấy tất cả sản phẩm
products = session.query(Product).all()
print("Danh sách sản phẩm:")
for product in products:
    print(product)

# Lấy sản phẩm theo ID
product = session.query(Product).filter_by(id=1).first()
if product:
    print(f"Sản phẩm có ID 1: {product}")


"""7. CẬP NHẬT DỮ LIỆU"""
# Lấy sản phẩm cần cập nhật
product = session.query(Product).filter_by(id=1).first()
if product:
    product.name = "Laptop Gaming"
    session.commit()
    print(f"Sản phẩm đã được cập nhật: {product}")

"""8. XÓA DỮ LIỆU"""

product = session.query(Product).filter_by(id=2).first()

if product:
    session.delete(product)
    session.commit()
    print(f"Đã xóa sản phẩm: {product}")