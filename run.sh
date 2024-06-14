#!/bin/bash

# Run setup script for Streamlit configuration
sh setup.sh

# Detect OS and run the appropriate server
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    # Unix-like OS, use gunicorn
    gunicorn app:app &
else
    # Windows OS, use waitress
    python -m waitress app:app &
fi

# Start the Streamlit server to serve your Streamlit app
streamlit run front_end.py
