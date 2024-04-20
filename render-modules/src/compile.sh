#!/bin/bash

# Exit script on error
set -e

# Define build directory name
BUILD_DIR="build"

# Check if build directory exists, if not, create it
if [ ! -d "$BUILD_DIR" ]; then
  echo "Creating build directory..."
  mkdir "$BUILD_DIR"
fi

# Change into the build directory
cd "$BUILD_DIR"

# Configure the project with CMake
# Add any configuration options you need
echo "Configuring project..."
cmake ..

# Build the project
# You can also specify the -j flag to parallelize the build process
echo "Building project..."
make

# Optionally, you can add custom commands after the build process
# For example, to run tests or install the project
# make test
# sudo make install

echo "Build completed successfully."