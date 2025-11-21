FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scenario_1120.py .

EXPOSE 5000
CMD ["python", "scenario_1120.py"]