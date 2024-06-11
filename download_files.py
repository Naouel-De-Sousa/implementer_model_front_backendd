import os
import requests

def download_file(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)

# URLs de vos fichiers sur GitHub (raw URLs)
csv_url = 'https://github.com/Naouel-De-Sousa/implementer_model_front_backendd/raw/master/sample_data_for_model.csv'
model_url = 'https://github.com/Naouel-De-Sousa/implementer_model_front_backendd/raw/master/models/mon_pipeline_complet.joblib'

# Chemins locaux où les fichiers seront enregistrés
csv_local_path = 'sample_data_for_model.csv'
model_local_path = 'models/mon_pipeline_complet.joblib'

# Téléchargez les fichiers
download_file(csv_url, csv_local_path)
download_file(model_url, model_local_path)