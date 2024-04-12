
import streamlit as st
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from joblib import load
from streamlit import components
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib
matplotlib.use('Agg')  # Ou 'Qt5Agg' ou un autre backend interactif
import base64
from PIL import Image
import io
import time


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

st.header('Présentation des features importance globale du modele')
image = 'C:\\Users\\naoue\\Documents\\OpenClassroomDataScientist\\projet_7_version_3\\shap_global.png'
st.image(image, caption='model global features importance ', alt = 'Présentation des features importance globale du modele')

######################### Formulaire pour saisir l'ID client

# Fonction pour afficher le résultat de la prédiction
def display_prediction_result(prediction):
    if prediction == 1:
        st.markdown("<h2 style='color: red;'>Risque de défaut détecté : Refus de crédit</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>Pas de risque de défaut détecté : Accord de crédit</h2>", unsafe_allow_html=True)

# Fonction pour afficher une barre de progression personnalisée
def custom_progress_bar(progress, color):
    progress_bar_html = f"""
    <div style='width: 100%;'>
        <div style='width: {progress}%; background-color: {color}; height: 24px; border-radius: 5px;'>
        </div>
    </div>
    """
    st.markdown(progress_bar_html, unsafe_allow_html=True)

# Initialisation d'une variable pour suivre si 'Prédire' a été cliqué sans entrée valide
if 'predict_clicked' not in st.session_state:
    st.session_state['predict_clicked'] = False

client_id_input = st.text_input('Entrez l\'ID client:', '', key='unique_client_id_input_key')

if 'prediction_result' in st.session_state and st.session_state['predict_clicked']:
    display_prediction_result(st.session_state['prediction_result'])


############# SHAP 

import streamlit.components.v1 as components

def display_html_file_in_streamlit(html_file_path):
    HtmlFile = open(html_file_path, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=600)

# Simuler un processus de prédiction
# def simulate_prediction():
#     progress_bar = st.progress(0)
#     for percent_complete in range(100):
#         time.sleep(0.05)  # Simuler un calcul en attente
#         progress_bar.progress(percent_complete + 1)
#     return True  


#################### Bouton pour les predictions

# Bouton pour déclencher la prédiction
if st.button('Prédire'):
    #prediction_result = simulate_prediction()
    st.session_state['predict_clicked'] = True
    if client_id_input:  # Assurez-vous que 'client_id_input' est défini quelque part dans votre code avant ce bloc
        try:
            client_id_int = int(client_id_input)
            st.session_state['selected_client_id'] = client_id_int
            # Envoyer la requête au backend Flask
            response = requests.post('http://localhost:5000/predict', json={'client_id': client_id_int})
            if response.status_code == 200:
                data = response.json()
                prediction = data['prediction']
                st.session_state['prediction_result'] = prediction
                display_prediction_result(prediction)
                
                # # Après avoir reçu le résultat de la prédiction, définissez la couleur de la barre de progression
                # if st.session_state['prediction_result'] == 1:  # Supposons que 1 signifie refus de crédit
                #     custom_color = "red"
                # else:
                #     custom_color = "green"
                
                # custom_progress_bar(100, custom_color)  # Affichez la barre de progression à 100% avec la couleur appropriée

                # Afficher l'image SHAP si disponible
                if 'shap_image' in data:
                    shap_image_base64 = data['shap_image']
                    shap_image = Image.open(io.BytesIO(base64.b64decode(shap_image_base64)))
                    st.image(shap_image, caption='SHAP Visualization')

            else:
                st.error('Une erreur est survenue lors de la prédiction.')
        except ValueError:
            st.error("L'ID client doit être un nombre entier.")
    else:
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
        # Préparation de l'espace pour les 3 graphiques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Création du plot pour la première variable numérique
            fig1, ax1 = plt.subplots()
            ax1.hist(df[variable_choice_1].dropna(), bins=50, label='Distribution générale')
            if include_client_info and 'selected_client_id' in st.session_state:
                client_id = st.session_state['selected_client_id']
                client_info = df[df['SK_ID_CURR'] == client_id]
                if not client_info.empty:
                    client_value_1 = client_info[variable_choice_1].iloc[0]
                    ax1.axvline(client_value_1, color='r', linestyle='--', label='Client sélectionné')
                    ax1.legend()
            ax1.set_ylabel('Fréquence')
            ax1.set_xlabel(variable_choice_1)
            ax1.set_title(f'Distribution de {variable_choice_1}')
            st.pyplot(fig1)
            st.caption("affichage des information concernat la variable selectionnée")
            plt.close()
        
        with col2:
            # Création du plot pour la deuxième variable numérique
            fig2, ax2 = plt.subplots()
            ax2.hist(df[variable_choice_2].dropna(), bins=50, label='Distribution générale')
            if include_client_info and 'selected_client_id' in st.session_state and not client_info.empty:
                client_value_2 = client_info[variable_choice_2].iloc[0]
                ax2.axvline(client_value_2, color='r', linestyle='--', label='Client sélectionné')
                ax2.legend()
            ax2.set_ylabel('Fréquence')
            ax2.set_xlabel(variable_choice_2)
            ax2.set_title(f'Distribution de {variable_choice_2}')
            st.pyplot(fig2)
            plt.close()
        
        with col3:
            # Création du plot combinant les deux variables
            fig3, ax3 = plt.subplots()
            ax3.scatter(df[variable_choice_1], df[variable_choice_2], alpha=0.5)
            if include_client_info and 'selected_client_id' in st.session_state and not client_info.empty:
                ax3.scatter(client_value_1, client_value_2, color='r')
                ax3.annotate('Client sélectionné', (client_value_1, client_value_2), textcoords="offset points", xytext=(0,10), ha='center')
            ax3.set_xlabel(variable_choice_1)
            ax3.set_ylabel(variable_choice_2)
            ax3.set_title(f'Relation entre {variable_choice_1} et {variable_choice_2}')
            st.pyplot(fig3)
            plt.close()



















