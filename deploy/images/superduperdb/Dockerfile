# syntax = docker/dockerfile:1.3
FROM jupyterhub/k8s-singleuser-sample:3.1.0

# Temporarily switch to the root user for installing packages and accounts
# ---------------
USER root

ENV NB_USER=superduper \
    NB_UID=1000 \
    HOME=/home/superduper

# Install some basic packages
# ---------------
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
        wget \
        curl \
        unzip \
        vim \
        procps \
        make \
        # requirement for nbgitpuller
        git \
   # Purge apt cache
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*


# Replace the default Jupyter user with a SuperDuper user
# ---------------
RUN deluser jovyan && adduser \
        --disabled-password \
        --gecos "SuperDuper User" \
        --uid ${NB_UID} \
        --home ${HOME} \
        --force-badname \
        ${NB_USER} \
    # Set path to support binaries from pip.
    && export PATH=/home/superduper/.local/bin:$PATH

WORKDIR ${HOME}
USER ${NB_USER}


# Install Python dependencies
# ---------------
RUN pip install --no-cache-dir \
    # SuperDuperDB
    superduperdb \
    # JupyterLab extensions \
    theme-darcula \
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