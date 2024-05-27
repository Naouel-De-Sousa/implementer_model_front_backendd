import os
import requests

def download_file(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)

# URLs de vos fichiers sur GitHub (raw URLs)
csv_url = 'https://github.com/Naouel-De-Sousa/implementer_model_front_backendd/blob/master/donn%C3%A9es_pour_model.csv'
model_url = 'https://github.com/Naouel-De-Sousa/implementer_model_front_backendd/blob/master/models/mon_pipeline_complet.joblib'

# Chemins locaux où les fichiers seront enregistrés
csv_local_path = 'données_pour_model.csv'
model_local_path = 'models/mon_pipeline_complet.joblib'

# Téléchargez les fichiers
download_file(csv_url, csv_local_path)
download_file(model_url, model_local_path)