FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY map/ ./map/

COPY log_data/ ./log_data/

COPY server_1128.py .

EXPOSE 5000
CMD ["python", "server_1128.py"]
