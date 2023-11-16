#!/bin/bash

echo "Adding mongo to /etc/hosts"
echo 127.0.0.1 mongodb | sudo tee -a /etc/hosts

echo "Adding vector-search to /etc/hosts"
echo 127.0.0.1 vector-search | sudo tee -a /etc/hosts

echo "Adding cdc to /etc/hosts"
echo 127.0.0.1 cdc | sudo tee -a /etc/hosts

echo "Adding dask scheduler to /etc/hosts"
echo 127.0.0.1 scheduler | sudo tee -a /etc/hosts

echo "Print hosts"
cat /etc/hosts