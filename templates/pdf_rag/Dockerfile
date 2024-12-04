FROM python:3.10

WORKDIR /app

COPY ./superduper .
COPY ./plugins .
COPY ./templates/pdf_rag templates/pdf_rag

RUN pip install -e . && pip install plugins/mongodb && pip install streamlit
RUN pip install --no-cache-dir ./templates/pdf_rag/requirements.txt

RUN superduper bootstrap ./templates/pdf_rag
RUN python3 add_data.py
RUN superduper apply pdf_rag

CMD ["streamlit", "run", "templates/pdf_rag/streamlit.py", "--server.runOnSave", "true"]
