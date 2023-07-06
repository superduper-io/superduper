MONGO_INIT_SH=./mongo_init.sh
MONGO_INITDB_USERNAME=testmongodbuser
MONGO_INITDB_PASSWORD=testmongodbpassword
MONGO_PORT=27018

init_replicaset_cmd="\
	rs.initiate( \
	{ \
		_id:\"rs0\", \
		version: 1, \
		members: [ \
		{ _id: 0, host : \"localhost:$MONGO_PORT\" }] \
	} \
	); \
	rs.status(); \
"

create_user_cmd="\
    db.getSiblingDB('admin').createUser( \
        { \
            user: \"$MONGO_INITDB_USERNAME\", \
            pwd: \"$MONGO_INITDB_PASSWORD\", \
            roles: [ { role: \"root\", db: \"admin\" } ] \
         } \
    ); \
    rs.status(); \
"
# replicate set initiate
echo "Checking mongo container..."
until mongosh --host sddb-mongodb --port $MONGO_PORT --eval "print(\"waited for connection\")"
do
    sleep 1
done

echo "Initializing replicaset..."
mongosh --host sddb-mongodb --port $MONGO_PORT --eval "$init_replicaset_cmd"

mongosh --host sddb-mongodb --port $MONGO_PORT  --eval "$create_user_cmd"
