
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#######################################################
#Title display
html_temp = """
<div style="background-color: tomato; padding:10px; border-radius:10px">
<h1 style="color: white; text-align:center">Risque de défaut de crédit à la consommation</h1>
</div>
<p style="font-size: 20px; font-weight: bold; text-align:center">Support de decision de credit </p>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Utiliser st.markdown pour colorier le titre
#st.markdown("<h1 style='color: blue;'>Risque de défaut de crédit à la consommation</h1>", unsafe_allow_html=True)


# header/subheader
st.header('Objectif du Dashboard')
#st.subheader('plot')

# Text
st.write("Ce dashboard offre une plateforme interactive permettant une analyse approfondie et une visualisation intuitive des profils clients.\n\nConçu pour être accessible aux non-experts en data science, il fournit des scores détaillés et des interprétations claires pour chaque client, enrichissant la compréhension sans nécessiter de connaissances techniques approfondies.")



######################### Formulaire pour saisir l'ID client

# Fonction pour afficher le résultat de la prédiction
def display_prediction_result(prediction):
    if prediction == 1:
        st.markdown("<h2 style='color: red;'>Risque de défaut détecté : Refus de crédit</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>Pas de risque de défaut détecté : Accord de crédit</h2>", unsafe_allow_html=True)

# Initialisation d'une variable pour suivre si 'Prédire' a été cliqué sans entrée valide
if 'predict_clicked' not in st.session_state:
    st.session_state['predict_clicked'] = False

client_id_input = st.text_input('Entrez l\'ID client:', '', key='unique_client_id_input_key')

if 'prediction_result' in st.session_state and st.session_state['predict_clicked']:
    display_prediction_result(st.session_state['prediction_result'])

if st.button('Prédire'):
    st.session_state['predict_clicked'] = True  # Marquer que l'utilisateur a tenté une prédiction
    if client_id_input:
        client_id_valid = True

        try:
            client_id_int = int(client_id_input)
        except ValueError:
            st.error("L'ID client doit être un nombre entier.")
            client_id_valid = False

        if client_id_valid:
            response = requests.post('http://localhost:5000/predict', json={'client_id': client_id_int})
            if response.status_code == 200:
                prediction = response.json()['prediction']

                # Stocker le résultat de la prédiction dans st.session_state
                st.session_state['prediction_result'] = prediction
                display_prediction_result(prediction)  # Afficher le résultat avec une fonction dédiée
                # Sauvegarder l'ID du client sélectionné dans st.session_state
                st.session_state['selected_client_id'] = client_id_int

        else:
            st.error('Une erreur est survenue lors de la prédiction.')
    else:
        # Afficher l'erreur seulement si 'Prédire' a été cliqué sans ID
        if st.session_state['predict_clicked']:
            st.error("Veuillez entrer un ID client.")



#################### Bouton pour récupérer toutes les données et afficher le plot
            
# Exemple de chargement des données (mettez à jour selon votre logique de chargement)
if 'data' not in st.session_state or st.button('Charger les données'):
    response = requests.get('http://localhost:5000/get-all-client-info')
    if response.status_code == 200:
        all_client_data = response.json()
        df = pd.DataFrame(all_client_data)
        st.session_state['data'] = df
    else:
        st.error("Impossible de récupérer les données.")

# Si les données sont chargées
if 'data' in st.session_state:
    df = st.session_state['data']
    # Choix de la variable pour le plot
    variable_choice = st.selectbox('Choisir une variable pour le plot:', df.columns)

    # Option pour inclure les informations du client sélectionné
    include_client_info = st.checkbox("Inclure les informations du client sélectionné dans le graphique")

     # Vérifier si la variable sélectionnée est de type numérique
    if not pd.api.types.is_numeric_dtype(df[variable_choice]):
        st.warning('Veuillez choisir une variable de type numérique.')
    else:
        # Création du plot pour les variables numériques
        fig, ax = plt.subplots()
        ax.hist(df[variable_choice].dropna(), bins=50,label='Distribution générale')  # Suppression des NaN pour éviter des erreurs de plot
        if include_client_info and 'selected_client_id' in st.session_state:
            # Récupération des données du client sélectionné
            client_id = st.session_state['selected_client_id']
            client_info = df[df['SK_ID_CURR'] == client_id]  # Assurez-vous que 'client_id' est la bonne clé
    
            if not client_info.empty:
                client_value = client_info[variable_choice].iloc[0]
                ax.axvline(client_value, color='r', linestyle='--', label='Client sélectionné')
                ax.legend()

        ax.set_ylabel('Fréquence')
        ax.set_xlabel(variable_choice)
        ax.set_title(f'Distribution de {variable_choice}')
        st.pyplot(fig)  # Affiche le plot dans Streamlit










 ############# shap

#if st.button('Analyse SHAP'):
    #client_id = st.text_input('Entrez l\'ID du client pour l\'analyse SHAP')
    #response = requests.get(f'http://localhost:5000/shap-analysis/{client_id}')
    #if response.status_code == 200:
     #   data = response.json()
      #  st.image(data['url'], caption='Analyse SHAP')
#else:
    #st.error('Erreur lors de la récupération de l\'analyse SHAP')

        