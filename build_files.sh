#!/bin/bash

echo "Building project..."

# Build tailwind
echo "Building Tailwind CSS..."
cd arcane_project
npm install
npm run build:css
cd ..

# Collect static files
echo "Collecting static files..."
python arcane_project/manage.py collectstatic --noinput

echo "Build finished." 