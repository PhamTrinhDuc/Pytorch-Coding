FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Build Docker image
# docker build -t fastapi-demo .
# Chạy container
# docker run -d -p 8000:8000 fastapi-demo
# Dừng container 
# docker stop fastapi-demo < or id container> 
# Kiểm tra docker đang chạy 
# docker ps 