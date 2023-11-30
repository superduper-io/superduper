#!/bin/bash


# Ensure mongodb entry
if grep -q '127.0.0.1.*mongodb' /etc/hosts
then
   echo "mongodb found";
else
  echo "****************************"
  echo -e "You need to update /etc/hosts with these lines:\n"
  echo "127.0.0.1 mongodb"
  echo "127.0.0.1 vector-search"
  echo "127.0.0.1 cdc"
  echo "127.0.0.1 scheduler"
  echo "****************************"
  exit 255
fi