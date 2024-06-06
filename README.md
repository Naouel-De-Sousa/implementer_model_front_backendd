"Projet 7" 



# My Credit APP

## Description
Cette application est une API de prédiction de crédit utilisant Flask pour le backend et Streamlit pour le frontend. L'API permet de prédire la probabilité de défaut de paiement d'un client en utilisant un modèle de machine learning.

## Table des Matières
1. Prérequis
2. Structure du Projet
3. Installation
4. Configuration
5. Exécution
6. Utilisation
7. Endpoints de l'API


## Prérequis
- Python 3.9 
- Flask
- Streamlit
- PythonAnywhere

## Structure du Projet

```
./
    .app.py.swp
    .gitattributes
    .gitignore
    app.py
    application_test.csv
    application_train.csv
    bureau.csv
    bureau_balance.csv
    categorical_columns.pkl
    credit_card_balance.csv
    custom_lightgbm.py
    data_drift.ipynb
    data_drift_report.html
    donn�es_pour_model.csv
    donn�es_sans_target.csv
    donn�es_trait�es.csv
    download_files.py
    dummy_roc_curve.png
    encoders.pkl
    front_end.py
    generate_readme.py
    HomeCredit_columns_description.csv
    importance_features.png
    imputer_cat.pkl
    imputer_num.pkl
    installments_payments.csv
    key_ssh
    lgbm_importances01.png
    mlflow.db
    model_scorring_one-Copy1.ipynb
    model_scorring_one.ipynb
    mon_pipeline_complet.joblib
    POS_CASH_balance.csv
    preprocessor.joblib
    previous_application.csv
    Procfile
    README.md
    requirements.txt
    roc_curve.png
    sample_submission.csv
    setup.sh
    shap_global.png
    submission_grid_search.csv
    submission_grid_search_rf.csv
    templates
    test_app.py
```

#  Backend Flask

# Installation
## Cloner le Répertoire

git clone https://github.com/Naouel-De-Sousa/implementer_model_front_backendd.git
cd votre-repo

## Créer un Environnement Virtuel

conda create --name third_env  
conda activate third_env    

python -m venv env
source env/bin/activate  # Sur Windows, utilisez `env\Scripts\activate`

## Installer les Dépendances

pip install -r requirements.txt


## Endpoints de l'API

### GET /
Retourne un message de bienvenue.

Réponse:

json

{
    "message": "Bienvenue sur l'API de prédiction de crédit !"
}

### GET /predict
Prédit la probabilité de défaut de paiement pour un client donné.

Paramètres:

client_id (int) : L'ID du client à prédire.
Réponse:

json

{
    "prediction": int,
    "shap_image": "image_base64",
    "feature_names": ["feature1", "feature2", ...],
    "features": [[val1, val2, ...]],
    "probability_of_default": float
}


### GET /get-all-client-info
Retourne les informations de tous les clients sous format de graphique.


json

[
    {
        "feature1": value1,
        "feature2": value2,
        ...
    },
    ...
]




# Frontend Streamlit

## Description
Le frontend de l'application utilise Streamlit pour fournir une interface utilisateur interactive permettant de visualiser les prédictions et les explications de modèle pour les clients.

## Fonctionnalités Principales
1. Titre et Objectif :
Affiche un titre et une description de l'objectif du tableau de bord.

2. Importance Globale des Caractéristiques :
Affiche une image représentant l'importance globale des caractéristiques du modèle.

3. Formulaire de Prédiction :
Permet aux utilisateurs d'entrer l'ID d'un client pour obtenir une prédiction.

4. Résultats de la Prédiction :
Affiche les résultats de la prédiction, la probabilité de remboursement, et un graphique SHAP expliquant la prédiction.

5. Affichage des Données Clients :
Permet de visualiser les données de tous les clients et de tracer des graphiques pour comparer les caractéristiques des clients.

## Utilisation
1. Lancer l'Application :
Assurez-vous que le backend Flask est en cours d'exécution.
Exécutez le frontend Streamlit avec la commande streamlit run frontend.py.

2. Interagir avec l'Interface :
Utilisez le formulaire pour entrer un ID client et obtenir des prédictions.
Visualisez les graphiques et les explications pour comprendre les prédictions du modèle.