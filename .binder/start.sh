#!/bin/bash
mkdir -p ~/.data/db
# Start mongodb server
nohup ~/.tools/mongodb/bin/mongod --dbpath ~/.data/db > .mongo_logs.txt 2>&1 &
sleep 5

# Restore database
~/.tools/mongotools/bin/mongorestore ~/dump
