#!/bin/bash

set -e

# Maximum number of retries
max_retries=100

for ((i=1; i<=""$max_retries""; i++)); do
  # Check if mongo-init has exited (indicating mongo initialization)
  # Count the number of lines in the result
  line_count=$(docker compose ps -qa --filter status=exited mongo-init | wc -l)

  # Check the number of lines
  if [ "$line_count" -eq 0 ]; then
    echo "Waiting for mongo to be initialized (Attempt $i)..."
    sleep 5
  elif [ "$line_count" -eq 1 ]; then
    echo "Mongo has been successfully initialized."
    exit 0
  else
    echo "Unexpected number of mongo-init instances: $line_count"
    exit 255
  fi
done

# If the maximum number of retries is reached
echo "Mongo initialization failed after $max_retries attempts."
exit 1