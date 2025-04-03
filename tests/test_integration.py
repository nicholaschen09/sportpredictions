import pytest
import pandas as pd
from soccer_scraper.scraper import SoccerMatchScraper
from soccer_predictor.predictor import SoccerPredictionSystem

def test_integration_scraper_to_predictor(tmp_path):
    """Test integration between scraper and predictor"""
    # 1. Scrape real data
    scraper = SoccerMatchScraper()
    scraper.scrape_fbref(season="2023-2024")
    
    # Save to CSV
    df = pd.DataFrame(scraper.data)
    csv_path = tmp_path / "test_matches.csv"
    df.to_csv(csv_path, index=False)
    
    # 2. Load and use predictor
    predictor = SoccerPredictionSystem()
    
    # Make prediction
    result = predictor.predict_match(
        home_team=df['home_team'].iloc[0],
        away_team=df['away_team'].iloc[0]
    )
    
    assert len(result) == 2
    assert all(isinstance(score, (int, float)) for score in result)