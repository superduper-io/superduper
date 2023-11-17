#!/bin/bash


# Ensure mongodb entry
if grep -q '127.0.0.1.*mongodb' /etc/hosts
then
   echo "mongodb found";
else
  echo "****************************"
  echo -e "You need to update /etc/hosts:\n"
  echo "echo 127.0.0.1 mongodb | sudo tee -a /etc/hosts"
  echo "****************************"
  exit 255
fi