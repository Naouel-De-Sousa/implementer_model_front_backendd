
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

#########################

        
# Formulaire pour saisir l'ID client

client_id_input = st.text_input('Entrez l\'ID client:', '', key='unique_client_id_input_key')

# set the prediction result
def display_prediction_result(prediction):
    if prediction == 1:
        st.markdown("<h2 style='color: red;'>Risque de défaut détecté : Refus de crédit</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>Pas de risque de défaut détecté : Accord de crédit</h2>", unsafe_allow_html=True)
if 'prediction_result' in st.session_state:
    display_prediction_result(st.session_state['prediction_result'])

# bouton predire

if st.button('Prédire'):
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
                # Ici, on suppose que votre API renvoie aussi 'feature_importances' avec la prédiction
                feature_importances = response.json().get('feature_importances', {})

                # Stocker le résultat de la prédiction dans st.session_state
                st.session_state['prediction_result'] = prediction
                display_prediction_result(prediction)  # Afficher le résultat avec une fonction dédiée

                if feature_importances:
                    # Afficher les informations d'importance des features
                    df_importances = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
                    df_importances = df_importances.sort_values('Importance', ascending=True)
                    
                    st.write("Importance des Features :")
                    st.bar_chart(df_importances.set_index('Feature')['Importance'])
            else:
                st.error('Une erreur est survenue lors de la prédiction.')
    else:
        st.error("Veuillez entrer un ID client.")




# boutton Lorsque l'utilisateur appuie sur le bouton pour récupérer les informations du client
if st.button('Obtenir les informations du client'):
    if client_id_input:
        try:
            # Convertir l'ID client en entier pour éviter les erreurs
            client_id_int = int(client_id_input)
        except ValueError:
            st.error("L'ID client doit être un nombre entier.")
            client_id_int = None

        if client_id_int is not None:
            # Envoyer une requête à l'endpoint `/get-client-info` de l'API Flask
            response = requests.post('http://localhost:5000/get-client-info', json={'client_id': client_id_int})
            
            if response.status_code == 200:
                # Si la requête est réussie, afficher les informations
                client_info = response.json()
                
                
                # Permettre à l'utilisateur de sélectionner une feature à visualiser
                feature = st.selectbox('Sélectionnez une feature à visualiser:', list(client_info.keys()))
                
                # Afficher la valeur pour la feature sélectionnée en fonction de son type
                if isinstance(client_info[feature], (int, float)):
                    # Pour les données numériques, créer un petit DataFrame et l'afficher sous forme de graphique
                    df = pd.DataFrame([client_info[feature]], columns=[feature])
                    st.bar_chart(df)
                else:
                    # Pour les autres types de données, les afficher comme texte
                    st.write(f"Valeur de {feature}: {client_info[feature]}")
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
