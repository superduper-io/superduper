# Use an official Ubuntu image as a parent image
FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && \
  apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  build-essential \
  git \
  sudo \
  pandoc \
  make \
  python3.11 \
  python3.11-venv

# Install Node.js 20 for Doc Site
RUN curl -sL https://deb.nodesource.com/setup_20.x | sudo -E bash - && \
  sudo apt-get install -y nodejs

# Set environment variable

# Create a virtual environment named .venv
RUN python3.11 -m venv .venv

ENV PATH="/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED 1

# Activate the virtual environment and install piptools
RUN python3.11 -m pip install --upgrade pip-tools

# Activate the virtual environment and install required Python packages
RUN python3.11 -m pip install pytest ipykernel jupyter black interrogate ruff ibis ibis-framework pydantic mypy 

# # Copy the pyproject.toml file
# COPY pyproject.toml ./

# # Activate the virtual environment and install development dependencies
# RUN python3.11 -m pip install -e '.[dev]'

# Set the default command to run when the container starts
CMD ["/bin/bash"]