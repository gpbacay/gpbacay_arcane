#!/bin/bash

# Exit on error
set -e

echo "Building project..."

# Install Python dependencies
pip install -r requirements.txt

# Go into the sub-project directory to handle Node.js dependencies
cd arcane_project

# Install Node.js dependencies and build CSS
npm install
npm run build:css

# Go back to the root directory
cd ..

# Collect static files
python arcane_project/manage.py collectstatic --noinput

echo "Build finished." 