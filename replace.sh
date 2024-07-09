#!/bin/bash

# Define the directories where you want to perform the search and replace
DIRECTORIES=(
    "./superduperdb"
    "./test"
    "./deploy"
)

# Define the string to search for and the string to replace it with
SEARCH_STRING="superduperdb"
REPLACE_STRING="superduper"
SEARCH_STRING_WEB="superduperdb.com"
REPLACE_STRING_WEB="superduper.io"
SEARCH_STRING_CAP="SuperDuperDB"
REPLACE_STRING_CAP="superduper.io"
SEARCH_STRING_BIG="SUPERDUPERDB"
REPLACE_STRING_BIG="SUPERDUPER"

# Iterate over each directory and perform the search and replace
for DIRECTORY in "${DIRECTORIES[@]}"; do
    if [ -d "$DIRECTORY" ]; then
        find "$DIRECTORY" -type f \( -name "*.py" -o -name "*.md" -o -name "*.ipynb" -o -name "*.toml" -o -name "*.yaml" \) | while read -r FILE; do
            if [ -f "$FILE" ]; then
                sed -i '' "s/$SEARCH_STRING_WEB/$REPLACE_STRING_WEB/g" "$FILE"
                sed -i '' "s/$SEARCH_STRING/$REPLACE_STRING/g" "$FILE"
                sed -i '' "s/$SEARCH_STRING_CAP/$REPLACE_STRING_CAP/g" "$FILE"
                sed -i '' "s/$SEARCH_STRING_BIG/$REPLACE_STRING_BIG/g" "$FILE"
                echo "Replaced in: $FILE"
            fi
        done
        echo "Replacement of '$SEARCH_STRING' with '$REPLACE_STRING' completed in all .py, .md, and .ipynb files in directory '$DIRECTORY'."
    else
        echo "Directory '$DIRECTORY' does not exist."
    fi
done

DIRECTORY="./docs"
find "$DIRECTORY" -type f \( -name "*.py" -o -name "*.md" -o -name "*.ipynb" -o -name "*.toml" -o -name "*.yaml" \) | while read -r FILE; do
    if [ -f "$FILE" ]; then
        sed -i '' "s/$SEARCH_STRING_WEB/$REPLACE_STRING_WEB/g" "$FILE"
        sed -i '' "s/$SEARCH_STRING/$REPLACE_STRING/g" "$FILE"
        sed -i '' "s/$SEARCH_STRING_BIG/$REPLACE_STRING_BIG/g" "$FILE"
        sed -i '' "s/$SEARCH_STRING_CAP/superduper/g" "$FILE"
        sed -i '' "s/<factory>/None/g" "$FILE"
        echo "Replaced in: $FILE"
    fi
done
echo "Replacement of '$SEARCH_STRING' with '$REPLACE_STRING' completed in all .py, .md, and .ipynb files in directory '$DIRECTORY'."

sed -i '' "s/$SEARCH_STRING/$REPLACE_STRING/g" "docs/sidebars.js"
sed -i '' "s/$SEARCH_STRING_WEB/$REPLACE_STRING_WEB/g" "docs/sidebars.js"
sed -i '' "s/$SEARCH_STRING_CAP/$REPLACE_STRING_CAP/g" "docs/sidebars.js"
sed -i '' "s/$SEARCH_STRING_BIG/$REPLACE_STRING_BIG/g" "docs/sidebars.js"

sed -i '' "s/$SEARCH_STRING/$REPLACE_STRING/g" "pyproject.toml"
sed -i '' "s/$SEARCH_STRING_WEB/$REPLACE_STRING_WEB/g" "pyproject.toml"
sed -i '' "s/$SEARCH_STRING_CAP/$REPLACE_STRING_CAP/g" "pyproject.toml"
sed -i '' "s/$SEARCH_STRING_BIG/$REPLACE_STRING_BIG/g" "pyproject.toml"

sed -i '' "s/$SEARCH_STRING/$REPLACE_STRING/g" "docs/docusaurus.config.js"
sed -i '' "s/$SEARCH_STRING_WEB/$REPLACE_STRING_WEB/g" "docs/docusaurus.config.js"
sed -i '' "s/$SEARCH_STRING_CAP/$REPLACE_STRING_CAP/g" "docs/docusaurus.config.js"
sed -i '' "s/$SEARCH_STRING_BIG/$REPLACE_STRING_BIG/g" "docs/docusaurus.config.js"

sed -i '' "s/$SEARCH_STRING/$REPLACE_STRING/g" "Makefile"
sed -i '' "s/$SEARCH_STRING_WEB/$REPLACE_STRING_WEB/g" "Makefile"
sed -i '' "s/$SEARCH_STRING_CAP/$REPLACE_STRING_CAP/g" "Makefile"
sed -i '' "s/$SEARCH_STRING_BIG/$REPLACE_STRING_BIG/g" "Makefile"

mv docs/static/img/superduperdb.gif docs/static/img/superduper.gif
mv docs/content/reusable_snippets/connect_to_superduperdb.md docs/content/reusable_snippets/connect_to_superduper.md
mv docs/content/reusable_snippets/connect_to_superduperdb.ipynb docs/content/reusable_snippets/connect_to_superduper.ipynb
mv superduperdb superduper