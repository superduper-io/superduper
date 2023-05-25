ARG PY_VERSION=3.11.3
FROM python:${PY_VERSION}-bullseye AS requirements

ENV PATH=/root/.local/bin:$PATH
RUN pip install -U pip

COPY build/requirements.ci.txt /tmp/requirements.txt
COPY dist /tmp/dist

RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

RUN pip install --user --find-links /tmp/dist superduperdb


FROM python:${PY_VERSION}-bullseye as server

ENV SERVICE_NAME="superduperdb"

RUN addgroup --gid 1001 $SERVICE_NAME && \
    adduser --gid 1001 --shell /bin/false --disabled-password --gecos "" --uid 1001 $SERVICE_NAME && \
    mkdir -p /var/log/$SERVICE_NAME && \
    chown $SERVICE_NAME:$SERVICE_NAME /var/log/$SERVICE_NAME

COPY --from=requirements --chown=$SERVICE_NAME /root/.local /home/$SERVICE_NAME/.local

WORKDIR /home/$SERVICE_NAME
USER $SERVICE_NAME
ENV PATH=/home/$SERVICE_NAME/.local/bin:$PATH


FROM server as jupyter

RUN pip install --user jupyter

CMD ["jupyter", "notebook", "--NotebookApp.token=''", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
