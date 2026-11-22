#!/bin/bash

# Define source and destination paths
SOURCE_DIR="$HOME/adam/ros-vision-service/results/experiments/model-stats/_summary"
DEST_DIR="."

# Copy all files from source to destination
cp "$SOURCE_DIR"/* "$DEST_DIR/"

# Print a success message
echo "Successfully copied summary from $SOURCE_DIR files to $DEST_DIR"