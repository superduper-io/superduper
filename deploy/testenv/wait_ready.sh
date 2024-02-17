#!/bin/bash

set -e

# Maximum number of retries
max_retries=100

# Wait for exit waits for a service in docker-compose to exit.
wait_for_exit() {
  service_name=$1

  for ((i=1; i<=""$max_retries""; i++)); do
    # Check if mongo-init has exited (indicating mongo initialization)
    # Count the number of lines in the result
    line_count=$(docker compose ps -qa --filter status=exited "${service_name}" | wc -l)

    # Check the number of lines
    if [ "$line_count" -eq 0 ]; then
      echo "Waiting for ${service_name} to be initialized (Attempt $i)..."
      sleep 5
    elif [ "$line_count" -eq 1 ]; then
      echo "${service_name} has been successfully initialized."
      return 0
    else
      echo "Unexpected number of ${service_name} instances: $line_count"
      return 255
    fi
  done

  # If the maximum number of retries is reached
  echo "${service_name} initialization failed after $max_retries attempts."
  return 1
}


# Define an array
service_list=("mongo-init" "vector-search-init" "cdc-init" "sandbox-init")

# Wait for all the services to become ready.
for service_name in "${service_list[@]}"; do
    wait_for_exit "${service_name}"
done