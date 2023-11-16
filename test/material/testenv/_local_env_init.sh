#!/bin/bash

set -eu

echo "Starting Mongo Server"
docker compose up mongodb mongo-init --detach --remove-orphans

echo "Waiting for Mongo to become ready"
./wait_ready.sh

echo "Map container names to localhost"
./set_hosts.sh

# Start everything from the root
cd ../../../

echo "Starting Dask Scheduler"
dask scheduler &

echo "Starting Dask Worker"
SUPERDUPERDB_DATA_BACKEND='mongodb://superduper:superduper@mongodb:27017/test_db' dask worker "tcp://localhost:8786" &

echo "Starting Vector Search Service"
SUPERDUPERDB_DATA_BACKEND='mongodb://superduper:superduper@mongodb:27017/test_db' python -m superduperdb vector-search &

echo "Starting Change Data Capture (CDC) Service"
SUPERDUPERDB_DATA_BACKEND='mongodb://superduper:superduper@mongodb:27017/test_db' python -m superduperdb cdc &
