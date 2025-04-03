import torch
from .model import DeepPredictor
from .preprocessor import AdvancedPreprocessor
import joblib

class SoccerPredictionSystem:
    def __init__(self, model_path='model.pth', preprocessor_path='preprocessor.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        self.load_model(model_path)
        self.load_preprocessor(preprocessor_path)
        
    def load_model(self, path):
        self.model = DeepPredictor(input_size=128)  # Adjust size based on your features
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()
        
    def load_preprocessor(self, path):
        self.preprocessor = joblib.load(path)
        
    def predict_match(self, home_team, away_team, home_form=None, away_form=None):
        """
        Predict the score for a single match
        """
        # Prepare input data
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_form': home_form if home_form is not None else 0.0,
            'away_form': away_form if away_form is not None else 0.0,
            # Add other features as needed
        }
        
        # Convert to DataFrame and preprocess
        X = self.preprocessor.transform(pd.DataFrame([match_data]))
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(X_tensor)
            
        home_score, away_score = prediction[0].cpu().numpy()
        return round(home_score), round(away_score)
    
    def predict_upcoming_matches(self, matches_df):
        """
        Predict scores for multiple upcoming matches
        """
        X = self.preprocessor.transform(matches_df)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()