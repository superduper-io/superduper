---
sidebar_position: 2
---

# Installation

There are two ways to get started:

- [A local `pip` installation on your system](#pip).
- [Running the `superduper` docker image](#docker-image).

## Pip

`superduper` is available on [PyPi.org](https://pypi.org/project/superduper/).

### Prerequisites

Before you begin the installation process, please make sure you have the following prerequisites in place:

#### Operating System

`superduper` is compatible with several Linux distributions, including:

- MacOS
- Ubuntu
- Debian

#### Python Ecosystem

If you plan to install superduper from source, you'll need the following:

- `python3.10` or `python3.11`
- `pip` 22.0.4 or later

Your experience with `superduper` on Linux may vary depending on your system and compute requirements.

### Installation

To install `superduper` on your system using `pip`, open your terminal and run the following command:

```bash
pip install superduper
```

This command will install `superduper` along with a minimal set of common dependencies required for running the framework.
If you would like to use the `superduper.ext` subpackages (e.g. `superduper.ext.openai`), you may build a requirements file
with this command:

```bash
python3 -m superduper requirements <list-of-extensions> > requirements.txt
```

For example, this builds a `requirements.txt` file for `openai` and `torch`:

```bash
python3 -m superduper requirements openai torch > requirements.txt
```

## Docker Image

#### Using Pre-built Images

If you prefer using Docker, you can pull a pre-built Docker image from Docker Hub and run it with Docker version 19.03 or later:

```bash
docker run -p 8888:8888 superduperio/superduper:latest
```

This command installs the base `superduper` image. If you want to run the ready-to-use examples, you'll need to download the required  dependencies at runtime. 


#### Building the image yourself

For more control, you can build the Docker images yourself from the latest GitHub version as follows:

Clone the code:

```bash
git clone --branch main --single-branch --depth 1 https://github.com/superduper/superduper.git
make build_superduper
```

#### Developer's docker image and environment

If you wish to use your local copy of code with a docker build, execute the following command:

```bash
make build_sandbox
```

You will see something like these lines in `docker images`:

```bash
REPOSITORY                    TAG             IMAGE ID       CREATED        SIZE
superduper/sandbox          latest          88ddab334d17   5 days ago     2.69GB
```