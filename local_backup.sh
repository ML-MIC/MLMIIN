#!/bin/bash

# Configuration
SOURCE_DIR="."    # Replace with your source folder path
DEST_DIR="./exclude/MLMIINprv/local_bkp" # Replace with your destination folder path
SUFFIX="*local"          # Suffix to look for

# Ensure destination directory exists
mkdir -p "$DEST_DIR"

# Recursively find files ending with 'local'
# Use -print0 to safely handle filenames with spaces
find "$SOURCE_DIR" -type f -regex ".*local\..*" -print0 | while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    
    if [[ -f "$DEST_DIR/$filename" ]]; then
        echo "WARNING: File skipped (already exists): $file"
    else
        cp "$file" "$DEST_DIR/"
        echo "Copied: $file"
    fi
done