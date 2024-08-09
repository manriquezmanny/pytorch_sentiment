#!/bin/bash
set -e

# Install necessary Python packages
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install nltk

# Create the nltk_data directory
mkdir -p ./nltk_data

# Download NLTK data
python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data')"