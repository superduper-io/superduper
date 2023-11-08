#!/bin/bash

set -eu

# Define the user and database information
USER="${SDDB_USER}"
PASSWORD="${SDDB_PASS}"
DATABASE_NAME="${SDDB_DATABASE}"
ROLE="dbOwner"  # Replace with the appropriate role you want

# replicate set initiate
echo "Checking mongo container..."
until mongosh --host mongodb --eval "print(\"waited for connection\")"
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
        { _id: 0, host: "localhost:27017"}
      ]
  }
  )
  rs.status()
EOF

# Create new users
echo "Creating ${USER}:${PASSWORD}/${DATABASE_NAME}"
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