FROM python:3.10-bullseye

COPY backend backend

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r backend/requirements.in

ENTRYPOINT ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
