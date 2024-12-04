FROM python:3.10

WORKDIR /app

COPY ./superduper superduper
COPY pyproject.toml pyproject.toml
COPY ./plugins plugins
COPY ./templates/pdf_rag templates/pdf_rag

RUN pip install . plugins/mongodb streamlit ipython jupyter -r ./templates/pdf_rag/requirements.txt
RUN chmod +x templates/pdf_rag/install.sh
RUN ./templates/pdf_rag/install.sh
    
CMD ["/bin/bash"]