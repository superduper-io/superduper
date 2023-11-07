#!/bin/bash
set -e

# Define MongoDB connection details
MONGO_HOST="localhost"
MONGO_PORT="27017"
MONGO_ROOT_USER="${MONGO_INITDB_ROOT_USERNAME}"
MONGO_ROOT_PASS="${MONGO_INITDB_ROOT_PASSWORD}"

# Define the user and database information
USER="${SDDB_USER}"
PASSWORD="${SDDB_PASS}"
DATABASE_NAME="${SDDB_DATABASE}"
ROLE="dbOwner"  # Replace with the appropriate role you want


# Initialize the replica set
mongosh "mongodb://${MONGO_HOST}:${MONGO_PORT}" <<EOF
  rs.initiate(
    {
      _id: "rs0",
      version: 1,
      members: [
        { _id: 0, host: "$MONGO_HOST:$MONGO_PORT" }
      ]
    }
  )
EOF

# Create a new 'superduper:superduper' user for 'test_db'
mongosh "mongodb://${MONGO_HOST}:${MONGO_PORT}" <<EOF
  use ${DATABASE_NAME}
  db.createUser(
    {
      user: "${USER}",
      pwd: "${PASSWORD}",
      roles: [ { role: "${ROLE}", db: "${DATABASE_NAME}" } ]
    }
  )
EOF
