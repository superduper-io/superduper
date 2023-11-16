#!/bin/bash


echo "Terminating Mongo Server"
docker compose down

echo "Terminating Dask "
pkill dask