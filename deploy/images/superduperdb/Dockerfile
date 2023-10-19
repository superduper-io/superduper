# syntax = docker/dockerfile:1.3
FROM jupyterhub/k8s-singleuser-sample:3.1.0

ARG SUPERDUPERDB_EXTRAS

ENV NB_USER=superduper \
    NB_UID=1000 \
    HOME=/home/superduper

# Temporarily switch to the root for privileged operations.
# ---------------
USER root

# Install system packages
# ---------------
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
        # Required for downloading code/data from the internet \
        wget curl unzip git \
        # DevOps
        vim procps make \
   # Purge apt cache
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*


# Replace the default Jupyter user with a SuperDuper user
# ---------------
RUN deluser jovyan \
    && adduser \
        --disabled-password \
        --gecos "SuperDuper User" \
        --uid ${NB_UID} \
        --home ${HOME} \
        --force-badname \
        ${NB_USER}


# Switch to SuperDuper user
# ---------------
WORKDIR ${HOME}
USER ${NB_USER}
ENV PATH="${HOME}/.local/bin:$PATH"


# Install Python dependencies
# ---------------
RUN pip install --no-cache-dir --no-cache -U \
    # SuperDuperDB
    superduperdb[${SUPERDUPERDB_EXTRAS}] \
    # JupyterLab extensions \
    theme-darcula \
    ipywidgets \
    # jupyterlab-lsp \
    # 'python-lsp-server[all]' \
    # Purge pip cache
    && pip cache purge


# Configure Jupyterlab extensions.
# ---------------
COPY --chown=superduper ./labextensions/@superduperdb ${HOME}/.local/share/jupyter/labextensions/@superduperdb
COPY --chown=superduper ./apputils-extension/themes.jupyterlab-settings ${HOME}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings


# Pre-Install SuperDuper examples
# ---------------
COPY ./download-examples.sh ./
RUN ./download-examples.sh && \
    rm ./download-examples.sh


# Replay Jupyter's entrypoint
# ---------------
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]