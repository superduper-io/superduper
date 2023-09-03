(replicasetup)=
# How to create a replica set in MongoDB

In order to use change data capture in SuperDuperDB together with MongoDB, it's necessary
to use a MongoDB replica set. (This is a feature of MongoDB in order to preserve performance of master node.)

Here's a step-by-step guide on how to set up such a replica set.

## Single Node Setup

1. **Start MongoDB Instances**: Start MongoDB on each server using the following command (replace `/data/db` with your data directory path):
   ```bash
   mongod --port 27017 --dbpath /data/db --replSet rs0
   ```
   
 2. **Initialize replica set**:  In the MongoDB shell,We will initialise replica set with one node.
     Replace the following values accordingly:
     MONGO_INITDB_USERNAME: yourusername
     MONGO_INITDB_PASSWORD: yourpassword
     MONGO_INITDB_DATABASE: yourinitdatabase
  ```bash
     # Initiate
	 rs.initiate({ _id: "rs0", version: 1,  members: [ { _id: 0, host:"localhost:27017" } ]})
	 # Add init db user and password
	 db.getSiblingDB('admin').createUser({ user: "MONGO_INITDB_USERNAME", pwd: "MONGO_INITDB_PASSWORD", roles: [ { role: "root", db: "admin" } ] });
 ```
2. **Create an user**: Use the above user and password to connect to mongodb
```bash
	mongosh --host localhost --port 27017  -u MONGO_INITDB_USERNAME -p MONGO_INITDB_PASSWORD --eval "db.getSiblingDB('test_db').createUser({ user: "MONGO_INITDB_USERNAME", pwd: "MONGO_INITDB_PASSWORD", roles: [ { role: "dbAdmin", db: "MONGO_INITDB_DATABASE" } ] });"
 ```
 3. Done!!

## Multi Node Setup

1. **Install MongoDB**: First, you need to install MongoDB.

2. **Create Data Directories**: Create separate data directories on each server where MongoDB will store its data. For example:
   - Server 1: /data/db
   - Server 2: /data/db
   - Server 3: /data/db

3. **Start MongoDB Instances**: Start MongoDB on each server using the following command (replace `/data/db` with your data directory path):
   ```bash
   mongod --port 27017 --dbpath /data/db --replSet rs0
   ```

4. **Connect to Primary Node**: Connect to one of the MongoDB instances. This will be the primary node, where you will initiate the replica set configuration.

5. **Initiate Replica Set Configuration**: In the MongoDB shell, run the following command to initiate the replica set configuration:
   ```bash
   # Initiate
   rs.initiate();
   # Add init db user and password
   db.getSiblingDB('admin').createUser({ user: "MONGO_INITDB_USERNAME", pwd: "MONGO_INITDB_PASSWORD", roles: [ { role: "root", db: "admin" } ]});
   # Create a test database with the init db user.
   mongosh --host localhost --port 27017  -u MONGO_INITDB_USERNAME -p MONGO_INITDB_PASSWORD --eval "db.getSiblingDB('test_db').createUser({ user: "MONGO_INITDB_USERNAME", pwd: "MONGO_INITDB_PASSWORD", roles: [ { role: "dbAdmin", db: "MONGO_INITDB_DATABASE" } ] });
   ```

6. **Add Secondary Nodes**: After running `rs.initiate()`, your primary node is set up. Now, you can add the secondary nodes to the replica set using the following command (replace `server2:27017`, `server3:27017`, etc., with the addresses of your secondary nodes):
   ```bash
   rs.add("server2:27017")
   rs.add("server3:27017")
   ```

7. **Check Replica Set Status**: To check the status of your replica set, use the following command in the MongoDB shell:
   ```bash
   rs.status()
   ```

8. **Configure Replica Set Options (Optional)**: You can further customize your replica set by modifying its configuration options, like setting priorities, hidden members, voting configuration, etc. Refer to the MongoDB documentation for more details.

## How to connect with superduperdb
```python
import pymongo
from superduperdb import superduper

client = pymongo.MongoClient("mongodb://<MONGO_INITDB_USERNAME>:<MONGO_INITDB_PASSWORD>@localhost:27017")
db = client.test_db

db = superduper(db)