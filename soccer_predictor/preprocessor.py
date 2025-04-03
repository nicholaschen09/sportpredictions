import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        # Convert categorical variables to numerical
        # Example features to consider:
        numerical_features = [
            'home_possession', 'away_possession',
            'home_shots', 'away_shots',
            'home_form', 'away_form',  # Last 5 matches
            'head_to_head_wins',
            'days_rest'
        ]
        
        X = df[numerical_features]
        X = self.scaler.fit_transform(X)
        
        # Target variables
        y = df[['home_score', 'away_score']].values
        
        return X, y