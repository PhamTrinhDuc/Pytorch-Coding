# Sử dụng image Python chính thức, phiên bản 3.11, phiên bản slim để giảm kích thước
FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements để cài đặt các dependency
# Làm điều này trước khi sao chép toàn bộ source code để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện Python
# --no-cache-dir giúp giảm kích thước image bằng cách không lưu cache
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ source code vào container
COPY . .

# Command mặc định để chạy ứng dụng khi container khởi động
# Sử dụng Uvicorn để chạy ứng dụng FastAPI
# Lắng nghe trên tất cả các địa chỉ IP và cổng 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



# Build Docker image:
# docker build -t "name_image" .

# Chạy container:
# docker run -d -p 8000:8000 "name_image"

# Dừng container:
# docker stop fastapi-demo < or id container> 

# Check Docker imges:
# docker images
    
# Check Docker Container:
# docker ps
# docker ps -a

# Xóa Docker Container:
# docker rm [CONTAINER ID]

# Xóa Docker Images:
# docker rmi [IMAGE ID]