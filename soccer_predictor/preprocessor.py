import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class AdvancedPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    def calculate_form(self, df, team, last_n=5):
        """Calculate team form based on last N matches"""
        df = df.sort_values('date')
        form_points = []
        
        for idx in range(len(df)):
            last_matches = df.iloc[max(0, idx-last_n):idx]
            points = 0
            matches_counted = 0
            
            for _, match in last_matches.iterrows():
                if match['home_team'] == team:
                    if match['home_score'] > match['away_score']:
                        points += 3
                    elif match['home_score'] == match['away_score']:
                        points += 1
                elif match['away_team'] == team:
                    if match['away_score'] > match['home_score']:
                        points += 3
                    elif match['home_score'] == match['away_score']:
                        points += 1
                matches_counted += 1
            
            form_points.append(points / matches_counted if matches_counted > 0 else 0)
            
        return form_points
    
    def add_features(self, df):
        """Add advanced features to the dataset"""
        # Add form for each team
        for team in pd.concat([df['home_team'], df['away_team']]).unique():
            df[f'{team}_form'] = self.calculate_form(df, team)
        
        # Add head-to-head statistics
        df['h2h_home_wins'] = 0
        df['h2h_away_wins'] = 0
        
        for idx, match in df.iterrows():
            previous_matches = df[
                ((df['home_team'] == match['home_team']) & (df['away_team'] == match['away_team'])) |
                ((df['home_team'] == match['away_team']) & (df['away_team'] == match['home_team']))
            ]
            previous_matches = previous_matches[previous_matches.index < idx]
            
            if len(previous_matches) > 0:
                home_wins = sum((previous_matches['home_team'] == match['home_team']) & 
                              (previous_matches['home_score'] > previous_matches['away_score']))
                away_wins = sum((previous_matches['away_team'] == match['away_team']) & 
                              (previous_matches['away_score'] > previous_matches['home_score']))
                df.at[idx, 'h2h_home_wins'] = home_wins
                df.at[idx, 'h2h_away_wins'] = away_wins
        
        return df
    
    def prepare_features(self, df):
        df = self.add_features(df)
        
        numerical_features = [
            'home_xg', 'away_xg',
            'h2h_home_wins', 'h2h_away_wins',
            'home_form', 'away_form'
        ]
        
        categorical_features = [
            'home_team', 'away_team'
        ]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
            ])
        
        # Prepare X and y
        X = preprocessor.fit_transform(df)
        y = df[['home_score', 'away_score']].values
        
        return X, y
