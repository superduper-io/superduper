#!/bin/bash


echo "Terminating Mongo Server"
docker compose down mongodb

echo "Terminating Dask "
pkill dask