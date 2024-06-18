#!/bin/bash

# Run setup script for Streamlit configuration
sh setup.sh

# Start the Flask app using gunicorn
gunicorn app:app -b 0.0.0.0:$PORT &

# Start the Streamlit app
streamlit run front_end.py --server.port 8501



