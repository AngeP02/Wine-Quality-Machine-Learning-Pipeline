FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY AppCloud.py ./
COPY Utils.py ./

CMD ["streamlit", "run", "AppCloud.py", "--server.port", "8080", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]