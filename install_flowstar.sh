#!/bin/bash

# Set the installation directory
INSTALL_DIR="/opt/flowstar"

# Create the installation directory if it doesn't exist
sudo mkdir -p $INSTALL_DIR

# Navigate to the installation directory
cd $INSTALL_DIR

# Download Flowstar
wget https://www.cs.colorado.edu/~xich8622/src/flowstar-2.1.0.tar.gz

# Extract the archive
tar -xf flowstar-2.1.0.tar.gz

# Navigate to the extracted directory
cd flowstar-2.1.0

# Install Flowstar dependencies
sudo apt-get install -y libgmp-dev libmpfr-dev libgsl-dev libglpk-dev bison flex gnuplot

# Build Flowstar
cd $INSTALL_DIR
cd flowstar-2.1.0
make

# Install Flowstar
cd $INSTALL_DIR
sudo cp flowstar-2.1.0/flowstar /usr/local/bin

# Cleanup
rm -rf flowstar-2.1.0.tar.gz 

echo "Flowstar has been successfully installed!"
