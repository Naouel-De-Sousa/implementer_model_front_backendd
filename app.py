from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import re
from joblib import load
import pdb
from flask_caching import Cache
from flask_cors import CORS
import matplotlib.pyplot as plt
import shap
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')  



app = Flask(__name__)
CORS(app)
# Configuration de Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  # Utilise un cache en mémoire

# charger traitement et modele 
pipeline = load('C:\\Users\\naoue\Documents\\OpenClassroomDataScientist\\projet_7_version_3\\models\\mon_pipeline_complet.joblib')

######################
@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction de crédit !"

# Remplacer les +inf/-inf par NaN
def replace_infinities(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data

def clean_feature_names(data):
    data.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in data.columns]
    return data

# Remplacer les caractères spéciaux par des underscores ou tout autre caractère de votre choix
def clean_feature_names_two(df):
    clean_names = {col: re.sub(r'[^\w\s]', '_', col) for col in df.columns}
    return df.rename(columns=clean_names)

#réduire la charge sur le serveur pour des requêtes répétées
@cache.memoize(timeout=300)  # Met en cache le résultat pour 300 secondes

# charger que les données utiles
def load_client_data(client_id, client_data_path):
    # Charger l'intégralité du fichier CSV en mémoire
    data = pd.read_csv(client_data_path)
    # Filtrer les données pour obtenir uniquement l'entrée correspondant à l'ID client spécifié
    filtered_data = data[data['SK_ID_CURR'] == client_id]
    return filtered_data

 



############## Fonctions de traitement des données
def preprocess_data(data):
    # Remplacer les infinis par NaN
    data = replace_infinities(data)
    print(data.columns)

    # Vérifier si 'SK_ID_CURR' est dans le DataFrame et le convertir en int
    if 'SK_ID_CURR' in data.columns:
        data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(float).astype(int)

    # Nettoyer les noms des caractéristiques
    data_cleaned = clean_feature_names(data)
    data_final = clean_feature_names_two(data_cleaned)
    return data_final




#####################prediction
@app.route('/predict', methods=['POST'])

def predict():
    content = request.json
    try:
        # Assurez-vous que client_id est un entier
        client_id = int(content['client_id'])
    except (ValueError, TypeError, KeyError):
        # Retourner une erreur si la conversion échoue ou si client_id est manquant
        return jsonify({'error': 'client_id doit être un entier'}), 400

    # Chemin vers votre fichier de données (ajustez selon votre configuration)
    client_data_path = 'C:\\Users\\naoue\\Documents\\OpenClassroomDataScientist\\Projet_7_version_3\\données_pour_model.csv'

    # Charger les données complètes du client
    client_data = load_client_data(client_id, client_data_path)
    
    # Si aucune donnée n'est trouvée pour le client_id donné, renvoyez une erreur
    if client_data.empty:
        return jsonify({'error': 'Client not found'}), 404

    # Prétraiter les données du client
    cleaned_data = preprocess_data(client_data)

    # Faire la prédiction avec le pipeline
    prediction = pipeline.predict(cleaned_data).tolist()  # Convertir en liste

    # Calculer les valeurs SHAP
    data_preprocessed = pipeline.named_steps['preprocessor'].transform(cleaned_data)
    explainer = shap.Explainer(pipeline.named_steps['classifier'])

    shap_values = explainer.shap_values(data_preprocessed)

    # Convertir les valeurs SHAP en liste pour une transmission facile via JSON
    shap_values_list = shap_values.tolist()

    # renvoyer les noms des caractéristiques
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    # Après avoir calculé les valeurs SHAP et les noms des caractéristiques
    results = {
        "prediction": int(prediction[0]),
        "shap_values": shap_values_list,
        "feature_names": feature_names.tolist()
    }

    # Retourner les résultats JSON
    return jsonify(results)





############### all client info
@app.route('/get-all-client-info', methods=['GET'])  # Utiliser GET puisqu'on ne soumet pas de données
def get_all_client_info():
    # Chemin vers votre fichier de données
    client_data_path = 'C:\\Users\\naoue\\Documents\\OpenClassroomDataScientist\\Projet_7_version_3\\données_pour_model.csv'
    
    # Charger les données complètes
    client_data = pd.read_csv(client_data_path)
    
    # Sélectionner les colonnes désirées, si nécessaire
    # Exemple : sélectionner les 20 premières colonnes
    client_data = client_data.iloc[:,:20]  # Optionnel, selon votre besoin
    
    # Convertir le DataFrame en dictionnaire pour le jsonify
    # Utiliser orient='records' pour obtenir une liste de dictionnaires
    data_dict = client_data.to_dict(orient='records')
    
    return jsonify(data_dict)



################SHAP





#####################
@app.route('/get-client-info', methods=['get'])

def get_client_info():
    content = request.json
    client_id = content['client_id']
    # Chemin vers votre fichier de données
    client_data_path = 'C:\\Users\\naoue\\Documents\\OpenClassroomDataScientist\\Projet_7_version_3\\données_pour_model.csv'
    # Utilisation de la fonction pour charger les données spécifiques du client
    client_info = load_client_data(client_id, client_data_path)
    if not client_info.empty:
        # Sélectionner les 20 premières colonnes
        client_info = client_info.iloc[:,:20]  # Utilise .iloc pour sélectionner les colonnes par position
        # Renvoyer les informations du client spécifiquement demandées
        return jsonify(client_info.to_dict(orient='records')[0])
    else:
        return jsonify({'error': 'Client not found'}), 404
    





if __name__ == '__main__':
    app.run(debug=True)