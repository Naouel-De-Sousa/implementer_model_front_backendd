import requests
from urllib.parse import quote as url_quote
import os
def download_file(url, local_path):
    # Encodage de l'URL pour gérer les caractères spéciaux
    encoded_url = url_quote(url, safe=':/')
    response = requests.get(encoded_url)
    
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

# Chemins locaux où les fichiers seront enregistrés
csv_local_path = os.path.abspath('./sample_data_for_model.csv')
model_local_path = './models/mon_pipeline_complet.joblib'

# Téléchargez les fichiers
download_file(csv_url, csv_local_path)
download_file(model_url, model_local_path)