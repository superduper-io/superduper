#!/bin/bash

set -eu

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


echo "Creating admin user: root@root/admin"
mongosh --host mongodb  <<EOF
    db.getSiblingDB('admin').createUser(
        {
            user: "root",
            pwd: "root",
            roles: [ { role: "root", db: "admin" } ]
         }
    )

    rs.status()
EOF

echo "Creating normal user: superduper:superduper/test_db"

USER="${SDDB_USER}"
PASSWORD="${SDDB_PASS}"
DATABASE_NAME="${SDDB_DATABASE}"
ROLE="dbOwner"  # Replace with the appropriate role you want


mongosh --host mongodb  <<EOF
  use ${DATABASE_NAME}
  db.createUser(
    {
      user: "${USER}",
      pwd: "${PASSWORD}",
      roles: [ { role: "${ROLE}", db: "${DATABASE_NAME}" } ]
    }
  )
EOF
