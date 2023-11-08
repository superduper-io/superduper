#!/bin/bash

set -eu

MONGO_INITDB_USERNAME=testmongodbuser
MONGO_INITDB_PASSWORD=testmongodbpassword

# replicate set initiate
echo "Checking mongo container..."
until mongosh --host mongodb  --eval "print(\"waited for connection\")"
do
    sleep 1
done

echo "Initializing replicaset..."
mongosh --host mongodb  <<EOF
    rs.initiate(
      {
          _id: "rs0",
          version: 1,
          members: [
            { _id: 0, host: "mongodb:27017"}
          ]
      }
    )
    rs.status()
EOF


# Create new admin user.
echo "Create user: ${MONGO_INITDB_USERNAME}:${MONGO_INITDB_PASSWORD}/admin"
mongosh --host mongodb  <<EOF
    db.getSiblingDB('admin').createUser(
        {
            user: "$MONGO_INITDB_USERNAME",
            pwd: "$MONGO_INITDB_PASSWORD",
            roles: [ { role: "root", db: "admin" } ]
         }
    )

    rs.status()
EOF
