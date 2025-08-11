FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY train.py /app/train.py

ENTRYPOINT ["python", "train.py"]
