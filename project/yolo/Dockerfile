FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY weights ./weights
COPY inference_server.py .

EXPOSE 8080
CMD ["python", "inference_server.py"]