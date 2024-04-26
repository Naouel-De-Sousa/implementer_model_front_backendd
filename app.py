import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import re
from joblib import load
import pdb
from flask_caching import Cache
from flask_cors import CORS
import shap
import matplotlib
import base64
import io
import requests


app = Flask(__name__)
CORS(app)
# Configuration de Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  # Utilise un cache en mémoire


# URL de votre modèle sur GitHub (lien direct/raw)
pipeline = 'https://github.com/Naouel-De-Sousa/implementer_model_front_backendd/raw/master/models/mon_pipeline_complet.joblib'

#pipeline = load('C:\\Users\\naoue\\Documents\\OpenClassroomDataScientist\\projet_7_version_3\\models\\mon_pipeline_complet.joblib')

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
    app.logger.debug(f"Received JSON: {request.json}")
    content = request.json
    try:
        # Assurez-vous que client_id est un entier
        client_id = int(content['client_id'])
    except (ValueError, TypeError, KeyError):
        # Retourner une erreur si la conversion échoue ou si client_id est manquant
        return jsonify({'error': 'client_id doit être un entier'}), 400

    # Chemin vers votre fichier de données (ajustez selon votre configuration)
    client_data_path = 'https://git-lfs.github.com/spec/v1'
    
    # Charger les données complètes du client
    client_data = load_client_data(client_id, client_data_path)
    
    # Si aucune donnée n'est trouvée pour le client_id donné, renvoyez une erreur
    if client_data.empty:
        return jsonify({'error': 'Client not found'}), 404

    # Prétraiter les données du client
    cleaned_data = preprocess_data(client_data)
    features_values = cleaned_data.values.tolist()

    # les predictions
    prediction = pipeline.predict(cleaned_data).tolist()
    probabilities = pipeline.predict_proba(cleaned_data)
    probability_of_default = probabilities[0][1] * 100  # calculer les proba
    
    expected_value = np.mean(prediction) # pour shap

    data_preprocessed = pipeline.named_steps['preprocessor'].transform(cleaned_data)
    
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()

    # Génération des valeurs SHAP et prédiction
    explainer = shap.Explainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(data_preprocessed)
    explanation = shap.Explanation(values=shap_values, data=data_preprocessed,base_values=expected_value, feature_names=feature_names)

    # Générez le plot SHAP (par exemple, un waterfall plot pour le premier échantillon)
    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=10)  # Modifier selon besoin
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encodez l'image en base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    results = {
        "prediction": int(prediction[0]),
        "shap_image": image_base64,  # Envoyez l'image encodée
        "feature_names": feature_names,
        "features": features_values ,
        "probability_of_default":probability_of_default
    }

    return jsonify(results)




############### all client info

@app.route('/get-all-client-info', methods=['GET'])  
def get_all_client_info():
    # Chemin vers fichier de données
    client_data_path = 'C:\\Users\\naoue\\Documents\\OpenClassroomDataScientist\\Projet_7_version_3\\données_pour_model.csv'
    
    # Charger les données complètes
    client_data = pd.read_csv(client_data_path)
    
    # Sélectionner les 20 premiere colonnes 
    client_data = client_data.iloc[:,:20]  # Optionnel, selon votre besoin
    
    # Convertir le DataFrame en dictionnaire pour le jsonify
    # Utiliser orient='records' pour obtenir une liste de dictionnaires
    data_dict = client_data.to_dict(orient='records')
    
    return jsonify(data_dict)



for rule in app.url_map.iter_rules():
    print(f'{rule.endpoint}: {rule}')



if __name__ == "__main__":
    app.run()


   