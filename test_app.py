
import pytest
from flask import json
from app import app, replace_infinities, clean_feature_names  # Adjust import as needed

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test the home endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Bienvenue" in response.data

def test_predict_no_data(client):
    """Test the predict endpoint without correct data, expects failure."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert b"client_id doit" in response.data

def test_replace_infinities():
    """Test the replace_infinities function."""
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        'a': [np.inf, -np.inf, 1, 2, 3]
    })
    df = replace_infinities(df)
    assert df.isna().sum().sum() == 2  # Expecting two NaNs for the infinities

def test_clean_feature_names():
    """Test the clean_feature_names function."""
    import pandas as pd
    df = pd.DataFrame(columns=['Test@Name', 'Another$Column'])
    df = clean_feature_names(df)
    assert all(col in df.columns for col in ['Test_Name', 'Another_Column'])

