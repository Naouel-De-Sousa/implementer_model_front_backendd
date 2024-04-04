
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from joblib import load
from streamlit import components
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')  # Ou 'Qt5Agg' ou un autre backend interactif
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


## SHAP 
## Fonction pour afficher le SHAP summary plot
def display_shap_summary(shap_values, feature_names):
    # Convertir les valeurs SHAP en un tableau numpy
    shap_values_array = np.array(shap_values)
    
    # Créer une explication SHAP avec les valeurs et les noms des caractéristiques
    explanation = shap.Explanation(values=shap_values_array, base_values=None, data=None, feature_names=feature_names)

    # Créer le SHAP summary plot avec le graphique beeswarm
    shap.summary_plot(explanation, plot_type="bar")


#################### Bouton pour les predictions
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
                data = response.json()
                prediction = data['prediction']

                shap_values = data.get('shap_values', [])
                feature_names = data.get('feature_names', [])

                # Stocker le résultat de la prédiction dans st.session_state
                st.session_state['prediction_result'] = prediction
                display_prediction_result(prediction)  # Afficher le résultat avec une fonction dédiée

                
                     # Afficher le SHAP summary plot si les valeurs SHAP sont disponibles
                    
                if shap_values and feature_names:
                    display_shap_summary(shap_values, feature_names) 
                # Sauvegarder l'ID du client sélectionné dans st.session_state
                st.session_state['selected_client_id'] = client_id_int


        else:
            st.error('Une erreur est survenue lors de la prédiction.')
    else:
        # Afficher l'erreur seulement si 'Prédire' a été cliqué sans ID
        if st.session_state['predict_clicked']:
            st.error("Veuillez entrer un ID client.")



#################### Bouton pour récupérer toutes les données et afficher le plot
            
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
    # Choix de la première variable pour le plot
    variable_choice_1 = st.selectbox('Choisir la première variable pour le plot:', df.columns, index=0)
    # Choix de la deuxième variable pour le plot
    variable_choice_2 = st.selectbox('Choisir la deuxième variable pour le plot:', df.columns, index=1)
    
    # Option pour inclure les informations du client sélectionné
    include_client_info = st.checkbox("Inclure les informations du client sélectionné dans le graphique")

    # Vérifier si les variables sélectionnées sont de type numérique
    if not pd.api.types.is_numeric_dtype(df[variable_choice_1]) or not pd.api.types.is_numeric_dtype(df[variable_choice_2]):
        st.warning('Veuillez choisir des variables de type numérique.')
    else:
        # Création du plot pour la première variable numérique
        fig, ax = plt.subplots()
        ax.hist(df[variable_choice_1].dropna(), bins=50, label='Distribution générale')
        if include_client_info and 'selected_client_id' in st.session_state:
            client_id = st.session_state['selected_client_id']
            client_info = df[df['SK_ID_CURR'] == client_id]
            if not client_info.empty:
                client_value_1 = client_info[variable_choice_1].iloc[0]
                ax.axvline(client_value_1, color='r', linestyle='--', label='Client sélectionné')
                ax.legend()
        ax.set_ylabel('Fréquence')
        ax.set_xlabel(variable_choice_1)
        ax.set_title(f'Distribution de {variable_choice_1}')
        st.pyplot(fig)

        # Option pour afficher le graphique combiné des deux variables
        if st.checkbox('Afficher le graphique combinant les deux variables sélectionnées'):
            fig, ax = plt.subplots()
            ax.scatter(df[variable_choice_1], df[variable_choice_2], alpha=0.5)
            if include_client_info and 'selected_client_id' in st.session_state:
                client_value_2 = client_info[variable_choice_2].iloc[0]
                ax.scatter(client_value_1, client_value_2, color='r')
                ax.annotate('Client sélectionné', (client_value_1, client_value_2), textcoords="offset points", xytext=(0,10), ha='center')
            ax.set_xlabel(variable_choice_1)
            ax.set_ylabel(variable_choice_2)
            ax.set_title(f'Relation entre {variable_choice_1} et {variable_choice_2}')
            st.pyplot(fig)





 ############# shap
