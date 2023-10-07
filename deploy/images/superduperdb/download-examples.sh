#!/bin/bash

set -eux

GITHUB_REPO=https://github.com/SuperDuperDB/superduperdb/tree/main/
BLOB_REPO=https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/
EXAMPLES_DIR=/docs/content/use_cases/items/

# Will download an index file container links to the documents within it.
wget -q --no-check-certificate ${GITHUB_REPO}/${EXAMPLES_DIR}

# Parse index and extract filepaths (keep only notebooks files ending with .ipynb).
# -v RS=',' sets the record separator (RS) to a comma. This means that awk will treat each comma-separated section of the JSON data as a separate record.
# -F'"' sets the field separator (FS) to double quotes ("), so awk will treat everything between double quotes as fields.
# /"path":/ is an awk pattern that looks for lines containing "path":.
# {print $4} prints the fourth field, which is the path value, for each matching line.
awk -v RS=',' -F'"' '/"path":/ {print $4}' index.html | grep ipynb &> ./file_list

# Download the notebook files to ./examples
for file in $(cat ./file_list); do
  echo "Downloading: ${BLOB_REPO}/${file}"
  wget  --directory-prefix ./examples  -q --no-check-certificate "${BLOB_REPO}/${file}"
done


rm ./index.html ./file_list
