#!/bin/bash

# Démarrer Streamlit en arrière-plan
streamlit run front_end.py &

# Démarrer Flask
gunicorn app:app --bind 0.0.0.0:$PORT --workers 3

