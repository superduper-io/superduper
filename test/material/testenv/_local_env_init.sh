#!/bin/bash

set -eux

echo "Starting Mongo Server"
docker compose up mongodb --detach --remove-orphans

echo "Setting host names"

echo "Initializing Mongo Server"
docker compose up mongo-init

echo "Waiting for Mongo to become ready"
./wait_ready.sh

# Start everything from the root
cd ../../../

echo "Starting Dask Scheduler"
dask scheduler &

echo "Starting Dask Worker"
SUPERDUPERDB_DATA_BACKEND='mongodb://superduper:superduper@localost:27017/test_db' dask worker "tcp://localhost:8786" &

echo "Starting Vector Search Service"
SUPERDUPERDB_DATA_BACKEND='mongodb://superduper:superduper@localost:27017/test_db' python -m superduperdb vector-search &

echo "Starting Change Data Capture (CDC) Service"
SUPERDUPERDB_DATA_BACKEND='mongodb://superduper:superduper@localost:27017/test_db' python -m superduperdb cdc
