FROM python:3.10.16
LABEL authors="sergio"

WORKDIR /app
COPY requirements.txt /app/
COPY model.onnx /app/
COPY main.py /app/

EXPOSE 5000

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["python3", "main.py"]