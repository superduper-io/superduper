# syntax = docker/dockerfile:experimental

# BUILD_ENV choses whether to use local source (sandbox) or released wheel (pypi).
ARG BUILD_ENV=pypi

# ---------------
# Configure Basic Template
# ---------------
FROM jupyterhub/k8s-singleuser-sample:3.1.0 as base

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
        # Required by OpenCV
        libglib2.0-0 libgl1-mesa-glx \
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


# Install common dependencies
# ---------------
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --upgrade \
    # JupyterLab extensions \
    theme-darcula \
    ipywidgets \
    jupyterlab-lsp \
    'python-lsp-server[all]' \
    "dask-labextension>=5"


# Install Jupyterlab extensions
# ---------------
COPY --chown=superduper ./images/superduperdb/labextensions/@superduperdb ${HOME}/.local/share/jupyter/labextensions/@superduperdb
COPY --chown=superduper ./images/superduperdb/apputils-extension/themes.jupyterlab-settings ${HOME}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings


# ---------------
# Configure Sandbox Build
# ---------------
FROM base AS build_sandbox
ONBUILD COPY --chown=superduper ./ ${HOME}/superduperdb
ONBUILD RUN cd ${HOME}/superduperdb; --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --upgrade \
    .[${SUPERDUPERDB_EXTRAS}]    \
    # Purge pip cache
    && pip cache purge


# ---------------
# Configure pypi Build
# ---------------
FROM base AS build_pypi
ONBUILD COPY --chown=superduper ${PWD}/examples ./examples
ONBUILD RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --upgrade \
    superduperdb[${SUPERDUPERDB_EXTRAS}] \
    # Purge pip cache
    && pip cache purge


# ---------------
# Select Build
# ---------------
FROM build_${BUILD_ENV}
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]