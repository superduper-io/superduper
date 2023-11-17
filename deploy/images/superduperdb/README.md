# SuperDuperDB Image

This image wraps the SuperDuperDB framework into a [Jupyter Notebook.](https://github.com/jupyterhub/zero-to-jupyterhub-k8s/blob/main/images/singleuser-sample/Dockerfile) 

Additionally, the image is shipped with a variety of examples from the SuperDuperDB [use-cases.](https://github.com/SuperDuperDB/superduperdb/tree/main//docs/content/use_cases/items)


To build the image: 

```shell
docker build -t superduperdb/superduperdb:latest   ./  --progress=plain
```


To run the image:

```shell
docker run -p 8888:8888 superduperdb/superduperdb:latest
``` 



## Building

For the sandbox we need to build the latest code against the latest dependencies.
This can be tricky as it requires for Dockerfile to access files outside of this directory.

Instead of moving the Dockerfile to the root directory, we use a trick with `Makefile`

.dockerconfig to ignore boring packets from sandbox

https://stackoverflow.com/questions/27068596/how-to-include-files-outside-of-dockers-build-context



https://stackoverflow.com/questions/31528384/conditional-copy-add-in-dockerfile