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
import os
import logging
import base64
from io import BytesIO
import lightgbm as lgb
from flask import abort
import git

app = Flask(__name__)
CORS(app)
# Configuration de Flask-Caching
# Utilise un cache en mémoire
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  
# Configurer la journalisation
logging.basicConfig(level=logging.DEBUG)



# Fonction pour télécharger un fichier depuis GitHub
def download_file_from_github(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        # Spécifiez le séparateur si nécessaire, par exemple, si le séparateur est ';' au lieu de ','
        return pd.read_csv(destination, sep=',', error_bad_lines=False, warn_bad_lines=True)  # Ignore les lignes mal formées
    else:
        raise Exception(f"Failed to download file from {url}")

# url direct
data_url = 'https://raw.githubusercontent.com/Naouel-De-Sousa/implementer_model_front_backendd/master/donn%C3%A9es_pour_model.csv'
model_url = 'https://raw.githubusercontent.com/Naouel-De-Sousa/implementer_model_front_backendd/master/models/mon_pipeline_complet.joblib'

# Chemins de destination locaux
data_path = os.path.abspath('./données_pour_model.csv')
model_dir = os.path.abspath('./models')
model_path = os.path.join(model_dir, 'mon_pipeline_complet.joblib')

# Créer le répertoire models s'il n'existe pas
os.makedirs(model_dir, exist_ok=True)

# Télécharger les fichiers de données
clients_df = download_file_from_github(data_url, data_path)

# Télécharger le modèle
logging.info(f"Téléchargement du modèle depuis {model_url}")
response = requests.get(model_url)
if response.status_code == 200:
    with open(model_path, 'wb') as file:
        file.write(response.content)
    logging.info(f"Modèle téléchargé avec succès et enregistré à {model_path}")
    try:
        pipeline = load(model_path)
        logging.info("Modèle chargé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        raise e
else:
    raise Exception(f"Failed to download file from {model_url}")


# Vérifier si 'SK_ID_CURR' est dans le DataFrame et le convertir en int
if 'SK_ID_CURR' in clients_df.columns:
    clients_df['SK_ID_CURR'] = clients_df['SK_ID_CURR'].astype(float).astype(int)
    clients_df.set_index('SK_ID_CURR', drop=False, inplace= True)

# URL de votre modèle sur GitHub (lien direct/raw)
#pipeline = load(os.path.abspath('./models/mon_pipeline_complet.joblib'))

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


############## Fonctions de traitement des données

def preprocess_data(data):
    # Remplacer les infinis par NaN
    data = replace_infinities(data)
    print(data.columns)

    # Nettoyer les noms des caractéristiques
    data_cleaned = clean_feature_names(data)
    data_final = clean_feature_names_two(data_cleaned)
    return data_final


#####################prediction 

@app.route('/predict', methods=['GET'])
def predict():
    
    app.logger.debug("Received request with arguments: %s", request.args)

    # Assurez-vous que client_id est un entier et présent dans les paramètres de l'URL
    try:
        client_id = int(request.args['client_id'])
    except (ValueError, KeyError):
        # Retourner une erreur si la conversion échoue ou si client_id est manquant
        return jsonify({'error': 'client_id doit être un entier et présent'}), 400


    # Charger les données complètes du client
    client_data = clients_df.loc[[client_id]]
    
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
    #explainer = shap.Explainer(pipeline.named_steps['classifier'])
 
    # Calculer les valeurs SHAP pour chaque prédiction
    explainer = shap.Explainer(pipeline.named_steps['classifier'], pipeline.named_steps['preprocessor'].transform(cleaned_data))
    shap_values = explainer.shap_values(data_preprocessed)


   # Créer un graphique SHAP
    shap.summary_plot(shap_values, features=features_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    # Enregistrer le graphique dans un buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convertir l'image en base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
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

    # Utiliser orient='records' pour obtenir un dictionnaires
    data_dict = clients_df.iloc[:,:20].to_dict(orient='records')

    return jsonify(data_dict)



for rule in app.url_map.iter_rules():
    print(f'{rule.endpoint}: {rule}')



if __name__ == "__main__":
    app.run()


   