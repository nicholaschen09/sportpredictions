import torch
import pytest
from soccer_predictor.model import SoccerPredictor

def test_model_creation():
    model = SoccerPredictor(input_size=8)
    assert isinstance(model, SoccerPredictor)

def test_model_forward_pass():
    model = SoccerPredictor(input_size=8)
    x = torch.randn(10, 8)  # batch of 10 samples, 8 features each
    output = model(x)
    assert output.shape == (10, 2)  # should output home and away scores

def test_model_training():
    X = torch.randn(100, 8)
    y = torch.randn(100, 2)
    model = SoccerPredictor(input_size=8)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Single training step
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    assert True  # If we get here, training step completed successfully