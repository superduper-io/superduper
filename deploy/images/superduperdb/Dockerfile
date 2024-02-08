# syntax = docker/dockerfile:experimental

# ---------------
# Global Parameters
# ---------------
# BUILD_ENV choses whether to use local source (sandbox) or released wheel (pypi).
ARG BUILD_ENV=release

# ---------------
# Configure Basic Template
# ---------------
FROM jupyterhub/k8s-singleuser-sample:3.1.0 as base

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
        # Required for PostgreSQL \
        libpq-dev \
	build-essential \
   # Purge apt cache
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*


# Replace the default Jupyter user with a SuperDuper user
# ---------------
ENV NB_USER=superduper \
    NB_UID=1000 \
    HOME=/home/superduper

RUN deluser jovyan && rm -rf /home/jovyan \
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
RUN python -m pip install setuptools pip
RUN --mount=type=cache,uid=1000,target=/home/superduper/.cache/pip python -m pip install --user \
    # JupyterLab extensions \
    theme-darcula \
    ipywidgets \
    jupyterlab-lsp \
    'python-lsp-server[all]'


# Install Jupyterlab extensions
# ---------------
COPY --chown=superduper ./deploy/images/superduperdb/labextensions/@superduperdb ${HOME}/.local/share/jupyter/labextensions/@superduperdb
COPY --chown=superduper ./deploy/images/superduperdb/apputils-extension/themes.jupyterlab-settings ${HOME}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings


# Volumes
# ---------------
VOLUME /artifacts

# Runtime Configuration
# ---------------

# Don't write .pyc files on the import of source modules.
ENV PYTHONDONTWRITEBYTECODE 1
# Set python cache directory
ENV PYTHONPYCACHEPREFIX "$HOME/.cache/cpython/"



# ---------------
# Build Sandbox
# ---------------
FROM base AS build_sandbox

ONBUILD ARG SUPERDUPERDB_EXTRAS=''
ONBUILD COPY --chown=superduper ./ ${HOME}/superduperdb
ONBUILD WORKDIR ${HOME}/superduperdb
# Install project dependencies.
ONBUILD RUN --mount=type=cache,uid=1000,target=/home/superduper/.cache/pip python -m pip install --user --editable .[${SUPERDUPERDB_EXTRAS}]
# Install user-defined dependencies
ONBUILD RUN --mount=type=cache,uid=1000,target=/home/superduper/.cache/pip python -m pip install --user  -r requirements.txt

# ---------------
# Build Release
# ---------------
FROM base AS build_release

ONBUILD ARG SUPERDUPERDB_EXTRAS=''
ONBUILD COPY --chown=superduper ${PWD}/examples ./examples
# Drop cache to reduce image size.
ONBUILD RUN python -m pip install --user superduperdb[${SUPERDUPERDB_EXTRAS}] \
    # Purge pip cache
    && pip cache purge
ONBUILD WORKDIR ${HOME}/examples

# ---------------
# Select Build
# ---------------
FROM build_${BUILD_ENV}
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
