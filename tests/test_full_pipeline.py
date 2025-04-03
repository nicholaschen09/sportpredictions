import pytest
import pandas as pd
import torch
import numpy as np
from soccer_scraper.scraper import SoccerMatchScraper
from soccer_predictor.preprocessor import AdvancedPreprocessor
from soccer_predictor.model import DeepPredictor
from soccer_predictor.predictor import SoccerPredictionSystem

@pytest.fixture
def sample_match_data():
    """Create sample match data for testing"""
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-08', '2023-01-15'],
        'home_team': ['Arsenal', 'Liverpool', 'Arsenal'],
        'away_team': ['Chelsea', 'Arsenal', 'Liverpool'],
        'home_score': [2, 1, 2],
        'away_score': [1, 1, 0],
        'home_xg': [2.1, 1.2, 1.8],
        'away_xg': [0.9, 1.1, 0.7],
    })

@pytest.fixture
def trained_model():
    """Create and return a small trained model for testing"""
    model = DeepPredictor(input_size=10)  # Small input size for testing
    return model

def test_scraper():
    """Test the scraper functionality"""
    scraper = SoccerMatchScraper()
    
    # Test FBRef scraping
    scraper.scrape_fbref(season="2023-2024")
    assert len(scraper.data) > 0
    
    # Check required fields
    required_fields = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    assert all(field in scraper.data[0] for field in required_fields)

def test_preprocessor(sample_match_data):
    """Test the preprocessing pipeline"""
    preprocessor = AdvancedPreprocessor()
    
    # Test form calculation
    form = preprocessor.calculate_form(sample_match_data, 'Arsenal')
    assert len(form) == len(sample_match_data)
    assert all(0 <= f <= 3 for f in form)  # Form should be between 0 and 3
    
    # Test feature preparation
    X, y = preprocessor.prepare_features(sample_match_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert y.shape[1] == 2  # Two outputs for home and away scores

def test_model_training(sample_match_data):
    """Test model training process"""
    preprocessor = AdvancedPreprocessor()
    X, y = preprocessor.prepare_features(sample_match_data)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    model = DeepPredictor(input_size=X.shape[1])
    
    # Test forward pass
    output = model(X_tensor)
    assert output.shape == y_tensor.shape
    
    # Test training step
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Single training step
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss)

def test_prediction_system(sample_match_data, tmp_path):
    """Test the full prediction system"""
    # Prepare test data and model
    preprocessor = AdvancedPreprocessor()
    X, y = preprocessor.prepare_features(sample_match_data)
    
    model = DeepPredictor(input_size=X.shape[1])
    
    # Save test model and preprocessor
    model_path = tmp_path / "test_model.pth"
    preprocessor_path = tmp_path / "test_preprocessor.pkl"
    
    torch.save(model.state_dict(), model_path)
    import joblib
    joblib.dump(preprocessor, preprocessor_path)
    
    # Test prediction system
    predictor = SoccerPredictionSystem(
        model_path=str(model_path),
        preprocessor_path=str(preprocessor_path)
    )
    
    # Test single match prediction
    home_score, away_score = predictor.predict_match(
        home_team="Arsenal",
        away_team="Chelsea",
        home_form=0.8,
        away_form=0.7
    )
    
    assert isinstance(home_score, (int, float))
    assert isinstance(away_score, (int, float))
    
    # Test batch prediction
    upcoming_matches = pd.DataFrame({
        'home_team': ['Arsenal', 'Liverpool'],
        'away_team': ['Chelsea', 'Manchester City'],
        'home_form': [0.8, 0.7],
        'away_form': [0.7, 0.9]
    })
    
    predictions = predictor.predict_upcoming_matches(upcoming_matches)
    assert len(predictions) == len(upcoming_matches)

def test_end_to_end():
    """Test the entire pipeline from scraping to prediction"""
    # 1. Scrape data
    scraper = SoccerMatchScraper()
    scraper.scrape_fbref(season="2023-2024")
    
    # 2. Process data
    preprocessor = AdvancedPreprocessor()
    df = pd.DataFrame(scraper.data)
    X, y = preprocessor.prepare_features(df)
    
    # 3. Train model
    model = DeepPredictor(input_size=X.shape[1])
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Single training step for testing
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    # 4. Make prediction
    test_input = torch.randn(1, X.shape[1])
    with torch.no_grad():
        prediction = model(test_input)
    
    assert prediction.shape == (1, 2)

if __name__ == "__main__":
    pytest.main([__file__])