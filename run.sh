#!/bin/bash

# Run setup script for Streamlit configuration
sh setup.sh

# Run the Flask app using Gunicorn
gunicorn -b 0.0.0.0:$PORT app:app &

# Start the Streamlit server to serve your Streamlit app
streamlit run front_end.py




