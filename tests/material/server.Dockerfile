ARG PY_VERSION=3.11.3
FROM python:${PY_VERSION} AS base
ENV PATH=/root/.local/bin:$PATH

WORKDIR /src

ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install -U --no-cache-dir pip

FROM base AS requirements

RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock ./

RUN poetry install --without dev

#RUN poetry export --without-hashes --with dev --format=requirements.txt -o requirements.txt
#RUN pip install -r requirements.txt

FROM requirements as server

COPY ./superduperdb .
COPY ./tests/material/server.py .

ENTRYPOINT ["python"]
CMD ["server.py"]