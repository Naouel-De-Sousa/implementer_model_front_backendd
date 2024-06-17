# Configurer joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from joblib import load
from flask_caching import Cache
from flask_cors import CORS
import shap
import matplotlib
import base64
import base64
from io import BytesIO
from urllib.parse import quote as url_quote
import requests


#from download_files import download_files  # Import the function

app = Flask(__name__)
CORS(app)
# Configuration de Flask-Caching
# Utilise un cache en mémoire
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  
#cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': '/home/Naouel/cache'})


# Download necessary files
#download_files()
#pd.set_option('future.no_silent_downcasting', True)# Appel du script pour télécharger les fichiers nécessaires
#os.system('python download_files.py')


def download_file(url, local_path):
    # Encodage de l'URL pour gérer les caractères spéciaux
    response = requests.get(url)
    # Vérification du statut de la réponse
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            file.write(response.content)
        print(f"Fichier téléchargé et enregistré sous {local_path}")
    else:
        print(f"Erreur lors du téléchargement du fichier depuis {url}. Statut: {response.status_code}")


# URLs de vos fichiers sur GitHub (raw URLs)
csv_url = 'https://github.com/Naouel-De-Sousa/implementer_model_front_backendd/raw/master/sample_data_for_model.csv'
model_url = 'https://github.com/Naouel-De-Sousa/implementer_model_front_backendd/raw/master/models/mon_pipeline_complet.joblib'




# Chemins de destination locaux
data_path = os.path.abspath('./sample_data_for_model.csv')
model_path = './models/mon_pipeline_complet.joblib'

# Téléchargez les fichiers
download_file(csv_url, data_path)
download_file(model_url, model_path)
clients_df = pd.read_csv(data_path)
pipeline = load(model_path)



# Vérifier si 'SK_ID_CURR' est dans le DataFrame et le convertir en int
if 'SK_ID_CURR' in clients_df.columns:
    clients_df['SK_ID_CURR'] = clients_df['SK_ID_CURR'].astype(float).astype(int)
    clients_df.set_index('SK_ID_CURR', drop=False, inplace= True)

app.logger.debug(f"DataFrame index: {clients_df.index}")
app.logger.debug(f"DataFrame head: {clients_df.head()}")
sample_client_ids = clients_df.index.tolist()[:5]
app.logger.debug(f"Sample client IDs: {sample_client_ids}")



######################
@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction de crédit !"


############## Fonctions de traitement des données
# Remplacer les +inf/-inf par NaN
def replace_infinities(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.infer_objects(copy=False)

def clean_feature_names(data):
    data.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in data.columns]
    return data

# Remplacer les caractères spéciaux par des underscores ou tout autre caractère de votre choix
def clean_feature_names_two(df):
    clean_names = {col: re.sub(r'[^\w\s]', '_', col) for col in df.columns}
    return df.rename(columns=clean_names)


#réduire la charge sur le serveur pour des requêtes répétées
@cache.memoize(timeout=300)  # Met en cache le résultat pour 300 secondes
def preprocess_data(data):
    # Remplacer les infinis par NaN
    data = replace_infinities(data)
    # Nettoyer les noms des caractéristiques
    data_cleaned = clean_feature_names(data)
    data_final = clean_feature_names_two(data_cleaned)

    return data_final



#####################prediction 


@app.route('/predict', methods=['GET'])
@cache.cached(timeout=600, query_string=True)
def predict():
    app.logger.debug("Received request with arguments: %s", request.args)

    try:
        client_id = int(request.args['client_id'])
        app.logger.debug(f"Converted client_id: {client_id}")
    except (ValueError, KeyError):
        return jsonify({'error': 'client_id doit être un entier et présent'}), 400

    if client_id not in clients_df.index:
        app.logger.debug(f"Client ID {client_id} not found in DataFrame index")
        return jsonify({'error': f'Client ID {client_id} not found'}), 404

    client_data = clients_df.loc[[client_id]]

    if client_data.empty:
        return jsonify({'error': 'Client not found'}), 404

    cleaned_data = preprocess_data(client_data)
    features_values = cleaned_data.values.tolist()

    try:
        app.logger.debug("Starting predict_proba")
        probabilities = pipeline.predict_proba(cleaned_data)
        app.logger.debug("Completed predict_proba")
    except Exception as e:
        app.logger.error(f"Error during predict_proba: {e}")
        return jsonify({'error': 'Prediction error'}), 500

    probability_of_default = probabilities[0][1] * 100

    try:
        app.logger.debug("Starting SHAP calculation")
        data_preprocessed = pipeline.named_steps['preprocessor'].transform(cleaned_data)
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()
        explainer = shap.Explainer(pipeline.named_steps['classifier'], data_preprocessed)
        shap_values = explainer(data_preprocessed)
        app.logger.debug("Completed SHAP calculation")
    except Exception as e:
        app.logger.error(f"Error during SHAP calculation: {e}")
        return jsonify({'error': 'SHAP calculation error'}), 500

    try:
        shap.summary_plot(shap_values, features=features_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
    except Exception as e:
        app.logger.error(f"Error during SHAP plot generation: {e}")
        return jsonify({'error': 'SHAP plot generation error'}), 500

    results = {
        "shap_image": image_base64,
        "feature_names": feature_names,
        "features": features_values,
        "probability_of_default": probability_of_default
    }

    return jsonify(results)


############### all client info

@app.route('/get-all-client-info', methods=['GET'])  
def get_all_client_info():

    # Utiliser orient='records' pour obtenir un dictionnaires
    data_dict = clients_df.iloc[:,:20].to_dict(orient='records')

    return jsonify(data_dict)


if __name__ == "__main__":
     app.run()



