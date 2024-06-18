#!/bin/bash

# Run setup script for Streamlit configuration
sh setup.sh

# Start the Flask server using Gunicorn
gunicorn app:app &

# Start the Streamlit server
streamlit run front_end.py


