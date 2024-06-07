# Configurer joblib
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Remplacez "4" par le nombre de cores logiques que vous souhaitez utiliser


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import re
from joblib import load
from joblib import Parallel, delayed, parallel_backend
import pdb
from flask_caching import Cache
from flask_cors import CORS
import shap
import matplotlib
import base64
import io
import requests
import logging
import base64
from io import BytesIO
import lightgbm as lgb
from flask import abort



pd.set_option('future.no_silent_downcasting', True)# Appel du script pour télécharger les fichiers nécessaires


os.system('python download_files.py')

app = Flask(__name__)
CORS(app)
# Configuration de Flask-Caching
# Utilise un cache en mémoire
#cache = Cache(app, config={'CACHE_TYPE': 'simple'})  
cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': '/home/Naouel/cache'})

@cache.memoize(timeout=600)  # Mettre en cache pour 600 secondes (10 minutes)
def some_long_running_function(i):
    # Simulation d'une opération longue
    import time
    time.sleep(10)
    return i
# Chemins de destination locaux
data_path = os.path.abspath('./données_pour_model.csv')
model_path = './models/mon_pipeline_complet.joblib'

clients_df = pd.read_csv(data_path)
# Debugging statement

pipeline = load(model_path)



# Vérifier si 'SK_ID_CURR' est dans le DataFrame et le convertir en int
if 'SK_ID_CURR' in clients_df.columns:
    clients_df['SK_ID_CURR'] = clients_df['SK_ID_CURR'].astype(float).astype(int)
    clients_df.set_index('SK_ID_CURR', drop=False, inplace= True)

app.logger.debug(f"DataFrame index: {clients_df.index}")
app.logger.debug(f"DataFrame head: {clients_df.head()}")
sample_client_ids = clients_df.index.tolist()[:5]
app.logger.debug(f"Sample client IDs: {sample_client_ids}")


# URL de votre modèle sur GitHub (lien direct/raw)
#pipeline = load(os.path.abspath('./models/mon_pipeline_complet.joblib'))

######################
@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction de crédit !"

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

# Assurez que client_id est un entier et présent dans les paramètres de l'URL
    try:
        client_id = int(request.args['client_id'])
        app.logger.debug(f"Converted client_id: {client_id}")
    except (ValueError, KeyError):
        return jsonify({'error': 'client_id doit être un entier et présent'}), 400
    
    # Vérifiez si le client_id existe dans l'index
    if client_id not in clients_df.index:
        app.logger.debug(f"Client ID {client_id} not found in DataFrame index")
        return jsonify({'error': f'Client ID {client_id} not found'}), 404

    # Charger les données complètes du client
    client_data = clients_df.loc[[client_id]]
    
    # Si aucune donnée n'est trouvée pour le client_id donné, renvoyez une erreur
    if client_data.empty:
        return jsonify({'error': 'Client not found'}), 404
    

    print('before preprocess')
    # Prétraiter les données du client
    cleaned_data = preprocess_data(client_data)
    print('after preprocess')
    features_values = cleaned_data.values.tolist()


    print('before predict')
    # les predictions
    prediction = pipeline.predict(cleaned_data).tolist()
    print('after predict')

    
    probabilities = pipeline.predict_proba(cleaned_data)
    print('after predict proba')
    probability_of_default = probabilities[0][1] * 100  # calculer les proba
    
    #expected_value = np.mean(prediction) # pour shap

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


