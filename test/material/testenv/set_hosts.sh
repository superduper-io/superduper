#!/bin/bash

echo "Adding mongo to /etc/hosts"
echo 127.0.0.1 mongodb | sudo tee -a /etc/hosts