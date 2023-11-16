#!/bin/bash


echo "Starting Mongo Server"
docker compose up mongodb mongo-init --detach --remove-orphans

echo "Starting Dask Scheduler"
dask scheduler &

echo "Starting Dask Worker"
SUPERDUPERDB_DATA_BACKEND='mongodb://superduper:superduper@mongodb:27017/test_db' dask worker "tcp://localhost:8786" &