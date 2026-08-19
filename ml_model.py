# ml_model.py - ML model logic with correct filename
import os
import logging
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        self.model = None
        # ✅ FIX: Use the correct filename
        self.model_path = 'house_price_model.pkl'  # Note: .pkl not .pk
        self.load_or_train()
    
    def load_or_train(self):
        """Load existing model or train a new one"""
        try:
            # Try to load existing model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ ML model loaded from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
        
        # Train new model
        try:
            logger.info("🔄 Training new ML model...")
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # ✅ Save with correct filename
            joblib.dump(self.model, self.model_path)
            logger.info(f"✅ ML model trained and saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to train model: {e}")
            return False
    
    def predict(self, features):
        """Make a prediction"""
        if self.model is None:
            return 400000.0  # Default fallback
        try:
            prediction = self.model.predict([features])[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 400000.0

# Singleton instance
ml_model = MLModel()