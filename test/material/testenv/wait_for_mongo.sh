#!/bin/bash

set -e

# Maximum number of retries
max_retries=10

for ((i=1; i<=""$max_retries""; i++)); do
  # Check if mongo-init has exited (indicating mongo initialization)
  result=$(docker compose ps -qa --status=exited mongo-init)

  # Count the number of lines in the result
  line_count=$(echo "$result" | wc -l)

  # Check the number of lines
  if [ "$line_count" -eq 0 ]; then
    echo "Waiting for mongo to be initialized (Attempt $i)..."
    sleep 3
  elif [ "$line_count" -eq 1 ]; then
    echo "Mongo has been initialized successfully."
    exit 0
  else
    echo "Unexpected number of mongo-init instances: $line_count"
    exit 255
  fi
done

# If the maximum number of retries is reached
echo "Mongo initialization failed after $max_retries attempts."
exit 1