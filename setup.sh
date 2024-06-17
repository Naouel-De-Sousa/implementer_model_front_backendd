#!/bin/bash

# Démarrer Streamlit en arrière-plan
streamlit run frontend/main.py &

# Démarrer Flask
gunicorn app:app --bind 0.0.0.0:$PORT --workers 3

