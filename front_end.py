
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from joblib import load
from streamlit import components
import base64
from PIL import Image
import io
import streamlit.components.v1 as components
import os



    
#######################################################
#Title display
html_temp = """
<div style="background-color: tomato; padding:10px; border-radius:10px">
<h1 style="color: white; text-align:center">Risque de défaut de crédit à la consommation</h1>
</div>
<p style="font-size: 20px; font-weight: bold; text-align:center">Support de decision de credit </p>
"""
st.markdown(html_temp, unsafe_allow_html=True)


# header/subheader
st.header('Objectif du Dashboard')
#st.subheader('plot')

# Text
st.write("Ce dashboard offre une plateforme interactive permettant une analyse approfondie et une visualisation intuitive des profils clients.\n\nConçu pour être accessible aux non-experts en data science, il fournit des scores détaillés et des interprétations claires pour chaque client, enrichissant la compréhension sans nécessiter de connaissances techniques approfondies.")

st.header('Présentation des features importance globale du modele')
image = Image.open(os.path.abspath('./shap_global.png'))

st.image(image, caption='model global features importance ')
st.text("Description alternative de l'image : Présentation des features importance globale du modèle.")

######################### Formulaire pour saisir l'ID client

# Fonction pour afficher le résultat de la prédiction et la probabilité
def display_prediction_result(prediction, probability_of_default):
    result_text = "Rembourse" if prediction == 0 else "Ne rembourse pas"
    probability_of_repayment = 100 - probability_of_default
    probability_text = f"Probabilité de remboursement : {probability_of_repayment:.2f}%"
    
    # Color based on prediction
    color = "green" if prediction == 0 else "red"

    # Display results
    st.write(f"Prédiction : {result_text}")
    st.markdown(f"**{probability_text}**", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{color};'>{probability_text}</h2>", unsafe_allow_html=True)
    
    # Préparer les données pour le bar chart
    states = ['Remboursement', 'Non-Remboursement']
    probabilities = [probability_of_repayment, probability_of_default]
    colors = ['green', 'red']  # Vert pour remboursement, Rouge pour non-remboursement
    
    # Créer le bar chart avec matplotlib
    fig, ax = plt.subplots()
    bars = ax.bar(states, probabilities, color=colors)
    ax.bar_label(bars)

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel('État')
    plt.ylabel('Probabilité (%)')
    plt.title('Probabilité de Remboursement vs Non-Remboursement')
    st.pyplot(fig)

def display_prediction_if_available():
    if 'prediction_result' in st.session_state and 'probability_of_default' in st.session_state:
        display_prediction_result(st.session_state['prediction_result'], st.session_state['probability_of_default'])


############## entrer client id 
client_id_input = st.text_input('Entrez l\'ID client:', '', key='unique_client_id_input_key')



def display_html_file_in_streamlit(html_file_path):
    HtmlFile = open(html_file_path, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=600)

#################### Bouton pour les predictions

# Bouton pour déclencher la prédiction
if st.button('Prédire'):
    if client_id_input:
        try:
            client_id_int = int(client_id_input)
            response = requests.get('http://naouel.pythonanywhere.com/predict',  params={'client_id': client_id_int})
            if response.status_code == 200:
                if response.text: # Vérifiez si la réponse n'est pas vide
                    data = response.json()
                    prediction = data['prediction']
                    probability_of_default = data.get('probability_of_default', 0)
                    st.session_state['prediction_result'] = prediction
                    st.session_state['probability_of_default'] = probability_of_default
                    display_prediction_result(prediction, probability_of_default)
                else:
                    st.error("La réponse de l'API est vide.")

                if 'shap_image' in data:
                    shap_image_base64 = data['shap_image']
                    shap_image = Image.open(io.BytesIO(base64.b64decode(shap_image_base64)))
                    st.image(shap_image, caption='SHAP Visualization')
            else:
                st.error('Une erreur est survenue lors de la prédiction. Code d\'erreur : {}'.format(response.status_code))
                st.text("Détails de l'erreur : " + response.text)

        except ValueError:
            st.error("L'ID client doit être un nombre entier.")
    else:
        st.error("Veuillez entrer un ID client.")

# Toujours réafficher les résultats de prédiction après chaque interaction si disponibles
display_prediction_if_available()

#################### Bouton pour récupérer toutes les données et afficher le plot

   
if client_id_input:
    try:
        st.session_state['selected_client_id'] = int(client_id_input)  # S'assurer que l'ID est un entier
    except ValueError:
        st.error("L'ID client doit être un nombre entier.")
else:
    st.warning("Veuillez entrer un ID client.")

   
if 'data' not in st.session_state or st.button('Charger les données'):
    response = requests.get('http://naouel.pythonanywhere.com/get-all-client-info')
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
            st.caption("affichage des information concernant la variable sélectionnée")
           
        
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

            plt.close('all')
            # Changement du backend
            plt.switch_backend('Agg')
    















