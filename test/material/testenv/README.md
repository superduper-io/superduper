# Run SuperDuperDB using Docker-Compose

#### Build and run your app with Compose
```shell
 docker-compose -f demo.yaml up 
```

### Available users:

`superduper:superduper` on `test_db` database

```shell
mongosh "mongodb://superduper:superduper@mongodb:27017/test_db"
```

Have in mind that `docker-compose` with source the `default credentials` from `.env` file.

#### Access Jupyter
Jupyter is now accessible by your browser:

```shell
http://localhost:8888/lab
```