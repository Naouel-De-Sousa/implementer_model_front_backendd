#!/bin/bash

# Run setup script for Streamlit configuration
sh setup.sh

# Start the Gunicorn server to serve your web application (Flask/Django)
gunicorn app:app &

# Start the Streamlit server to serve your Streamlit app
streamlit run front_end.py