# SuperDuperDB notebook examples

## Running with docker

To run these notebooks with `docker`, in the root directory, build the image with:

```bash
make testenv_image SUPERDUPERDB_EXTRAS=sandbox
```

Then start Jupyter with:

```bash
docker run -it -p 8888:8888 -v <path-to-parent>/superduperdb:/home/superduper/superduperdb superduperdb/sandbox
```