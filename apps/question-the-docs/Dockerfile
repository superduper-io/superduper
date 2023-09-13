FROM python:3.10-bullseye

WORKDIR /app

COPY ./.default_repo_config.json /app/.default_repo_config.json

COPY backend backend

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r backend/requirements.in

CMD ["uvicorn", "backend.main:app", "--forwarded-allow-ips='*'", "--host", "0.0.0.0", "--port", "8080"]
