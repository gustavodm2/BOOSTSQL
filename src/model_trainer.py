import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from tqdm import tqdm

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        self.best_model = None
        self.feature_extractor = None
    
    def train_models(self, X, y):
        
        results = {}
        n_splits = min(5, len(X) // 10000)
        
        for name, model in self.models.items():
            print(f"Treinando {name}...")
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
            
            model.fit(X, y)
            y_pred = model.predict(X)
            
            results[name] = {
                'model': model,
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred),
                'cv_mae_mean': -cv_scores.mean(),
                'cv_mae_std': cv_scores.std()
            }
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        
        print("\n=== RESULTADOS ===")
        for name, metrics in results.items():
            print(f"\n{name.upper():<15}")
            print(f"MAE: {metrics['mae']:.2f} ms")
            print(f"RMSE: {metrics['rmse']:.2f} ms") 
            print(f"R²: {metrics['r2']:.3f}")
            print(f"CV MAE: {metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f} ms")
        
        print(f"\nMELHOR MODELO: {best_model_name}")
        return results
    
    def save_model(self, filepath):
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'feature_extractor': self.feature_extractor
            }, filepath)
            print(f"💾 Modelo salvo: {filepath}")
    
    def load_model(self, filepath):
        loaded = joblib.load(filepath)
        self.best_model = loaded['model']
        self.feature_extractor = loaded['feature_extractor']
        return self