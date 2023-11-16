#!/bin/bash


echo "Starting Mongo Server"
docker compose up mongodb mongo-init --detach --remove-orphans

echo "Starting Dask Scheduler"
dask scheduler &

echo "Starting Dask Worker"
dask worker "tcp://localhost:8786" &