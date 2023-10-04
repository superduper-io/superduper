#!/bin/bash
set -e

mongosh "mongodb://${MONGO_INITDB_ROOT_USERNAME}:${MONGO_INITDB_ROOT_PASSWORD}@localhost" <<EOF
  use admin
  db.createUser(
    {
      user: "${SDDB_USER}",
      pwd: "${SDDB_PASS}",
      roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
    }
  )
EOF