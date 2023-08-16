FROM python:3.10-bullseye

WORKDIR /app

COPY backend backend

COPY frontend/ .

RUN apt-get update && apt-get install -y

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && apt-get install -y nodejs

RUN npm install && npm run build

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r backend/requirements.in

CMD ["uvicorn", "backend.main:app", "--forwarded-allow-ips='*'", "--host", "0.0.0.0", "--port", "8080"]
