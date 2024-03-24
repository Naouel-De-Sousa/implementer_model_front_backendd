
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

#st.title('Insights Client: Scores et Synthèses')

# header/subheader
st.header('Objectif du Dashboard')
#st.subheader('plot')

# Text
st.write("Ce dashboard offre une plateforme interactive permettant une analyse approfondie et une visualisation intuitive des profils clients.\n\nConçu pour être accessible aux non-experts en data science, il fournit des scores détaillés et des interprétations claires pour chaque client, enrichissant la compréhension sans nécessiter de connaissances techniques approfondies.")
# Plot

#Number of loans in the sample
#########################

        
# Formulaire pour saisir l'ID client
client_id_input = st.text_input('Entrez l\'ID client:', '')

# Lorsque l'utilisateur appuie sur le bouton 'Prédire'
if st.button('Prédire'):
    if client_id_input:
        # Envoyer une requête à l'API Flask
        response = requests.post('http://localhost:5000/predict', json={'client_id': client_id_input})
        
        if response.status_code == 200:
            # Afficher la prédiction
            prediction = response.json()['prediction']
            if prediction == 1:
                st.write("Risque de défaut détecté : Refus de crédit")
            else:
                st.write("Pas de risque de défaut détecté : Accord de crédit")
        else:
            st.error('Une erreur est survenue lors de la prédiction.')
    else:
        st.error("Veuillez entrer un ID client.")


# Lorsque l'utilisateur appuie sur le bouton pour récupérer les informations du client
if st.button('Obtenir les informations du client'):
    if client_id_input:
        # Envoyer une requête à l'endpoint `/get-client-info` de l'API Flask
        response = requests.post('http://localhost:5000/get-client-info', json={'client_id': int(client_id_input)})
        
        if response.status_code == 200:
            # Si la requête est réussie, afficher les informations
            client_info = response.json()
            st.write("Informations du client :")
            st.json(client_info)  # Affichage des informations du client sous forme de JSON dans l'interface
        else:
            st.error("Client non trouvé ou erreur dans l'API.")
    else:
        st.error("Veuillez entrer un ID client.")
        
        # ## features importance
    # # Imprimer les étapes du pipeline
    #for name, model in lgbm_object.named_steps.items():
    #     print(name, model)

    # lgbm_model = lgbm_object.named_steps['classifier']

    # feature_importances = lgbm_model.feature_importances_


    # df_feature_importances = pd.DataFrame({
    #     'Feature': feature_names,  # Assurez-vous que cette liste correspond à votre ensemble de données après transformation
    #     'Importance': feature_importances
    # }).sort_values('Importance', ascending=False)

    # # Affichage du graphique
    # plt.figure(figsize=(10, 6))
    # plt.barh(df_feature_importances['Feature'][:10], df_feature_importances['Importance'][:10], color='skyblue')
    # plt.xlabel('Importance')
    # plt.ylabel('Feature')
    # plt.title('Top 10 Feature Importances')
    # plt.gca().invert_yaxis()
    # plt.tight_layout()

    # st.pyplot(plt)

    # # Vérifier si le nombre d'importances correspond au nombre de noms de caractéristiques
    # print('this is an error')
    # assert len(feature_importances) == len(feature_names)
