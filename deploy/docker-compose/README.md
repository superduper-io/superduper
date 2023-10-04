# Run SuperDuperDB using Docker-Compose


#### Build and run your app with Compose
```shell
 docker-compose -f demo.yaml up 
```

Have in mind that `docker-compose` with source the `default credentials` from `.env` file.

#### Access Jupyter
Jupyter's access mechanism is based on the concept of authentication tokens. 
These tokens are normally written in the logs of `docker-compose`, but some times it can get messy.

To get the token directly, use:
```shell
docker logs docker-compose-superduperdb-1 |& grep -m 2 "/lab?token="
```