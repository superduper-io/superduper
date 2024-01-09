---
sidebar_position: 2
---

# Installation

There are two ways to get started:

- [A local `pip` installation on your system](#pip).
- [Running the `superduperdb` docker image](#docker-image).

## Pip

`superduperdb` is available on [PyPi.org](https://pypi.org/project/superduperdb/).

### Prerequisites

Before you begin the installation process, please make sure you have the following prerequisites in place:

#### Operating System

`superduperdb` is compatible with several Linux distributions, including:

- MacOS
- Ubuntu
- Debian

#### Python Ecosystem

If you plan to install SuperDuperDB from source, you'll need the following:

- `python3.10` or `python3.11`
- `pip` 22.0.4 or later

Your experience with `superduperdb` on Linux may vary depending on your system and compute requirements.

### Installation

To install `superduperdb` on your system using `pip`, open your terminal and run the following command:

```bash
pip install superduperdb
```

This command will install `superduperdb` along with a minimal set of  common dependencies required for running the framework. Some larger  dependencies, like `pytorch`, are not included to keep the image size small. You can install such dependencies using the following syntax:

```bash
pip install superduperdb[<category>]
```

Here are the available categories you can use:

- `api`: Installs clients for third-party services like OpenAI, Cohere, and Anthropic.
- `torch`: Installs PyTorch dependencies.
- `docs`: Installs tools for rendering Markdown files into websites.
- `quality`: Installs tools for aiding in the development of high-quality code.
- `testing`: Installs tools for testing the SuperDuperDB ecosystem.
- `dev`: Installs all the above categories.
- `demo`: Installs all the common dependencies and the dependencies required for the `examples`.

You can find more details on these categories in the [pyproject.toml](https://github.com/SuperDuperDB/superduperdb/blob/main/pyproject.toml) file.

## Docker Image

#### Using Pre-built Images

If you prefer using Docker, you can pull a pre-built Docker image from Docker Hub and run it with Docker version 19.03 or later:

```bash
docker run -p 8888:8888 superduperdb/superduperdb:latest
```

This command installs the base `superduperdb` image. If you want to run the ready-to-use examples, you'll need to download the required  dependencies at runtime. Alternatively, we provide a pre-built image  with all the dependencies for examples preinstalled:

```bash
docker run -p 8888:8888 superduperdb/demo:latest
```

#### Building the image yourself

For more control, you can build the Docker images yourself using the following commands:

```bash
make build_superduperdb
make build_demo
```