FROM python:3.10.16
LABEL authors="sergio"

WORKDIR /app
COPY requirements.txt /app/
COPY model.onnx /app/
COPY labelEncoder1.pkl /app/
COPY labelEncoder3.pkl /app/
COPY main.py /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["python3", "main.py"]