import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from tqdm import tqdm
from typing import Dict

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

class ModelTrainer:

    def __init__(self, config=None, feature_extractor=None):
        self.config = config
        self.feature_extractor = feature_extractor
        self.models = {}
        self.best_model = None
        self._initialize_models()

    def _initialize_models(self):
        
        self.models['linear_regression'] = LinearRegression()
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=self.config.ml_training['models']['random_forest']['n_estimators'] if self.config else 100,
            max_depth=self.config.ml_training['models']['random_forest']['max_depth'] if self.config else None,
            random_state=self.config.ml_training['models']['random_forest']['random_state'] if self.config else 42,
            n_jobs=-1
        )
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=self.config.ml_training['models']['gradient_boosting']['n_estimators'] if self.config else 100,
            learning_rate=self.config.ml_training['models']['gradient_boosting']['learning_rate'] if self.config else 0.1,
            max_depth=self.config.ml_training['models']['gradient_boosting']['max_depth'] if self.config else 3,
            random_state=self.config.ml_training['models']['gradient_boosting']['random_state'] if self.config else 42
        )

        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=self.config.ml_training['models']['xgboost']['n_estimators'] if self.config else 100,
                learning_rate=self.config.ml_training['models']['xgboost']['learning_rate'] if self.config else 0.1,
                max_depth=self.config.ml_training['models']['xgboost']['max_depth'] if self.config else 3,
                random_state=self.config.ml_training['models']['xgboost']['random_state'] if self.config else 42,
                n_jobs=-1
            )

        if HAS_LIGHTGBM:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=self.config.ml_training['models']['lightgbm']['n_estimators'] if self.config else 100,
                learning_rate=self.config.ml_training['models']['lightgbm']['learning_rate'] if self.config else 0.1,
                max_depth=self.config.ml_training['models']['lightgbm']['max_depth'] if self.config else 3,
                random_state=self.config.ml_training['models']['lightgbm']['random_state'] if self.config else 42,
                n_jobs=-1
            )

    def train_models(self, X, y):
        results = {}
        n_splits = max(2, min(5, len(X) // 100))
        for name, model in self.models.items():
            print(f'Treinando {name}...')
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
            model.fit(X, y)
            y_pred = model.predict(X)
            results[name] = {'model': model, 'mae': mean_absolute_error(y, y_pred), 'rmse': np.sqrt(mean_squared_error(y, y_pred)), 'r2': r2_score(y, y_pred), 'cv_mae_mean': -cv_scores.mean(), 'cv_mae_std': cv_scores.std()}
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        print('\n=== RESULTADOS ===')
        for name, metrics in results.items():
            print(f'\n{name.upper():<15}')
            print(f'MAE: {metrics['mae']:.2f} ms')
            print(f'RMSE: {metrics['rmse']:.2f} ms')
            print(f'R²: {metrics['r2']:.3f}')
            print(f'CV MAE: {metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f} ms')
        print(f'\nMELHOR MODELO: {best_model_name}')
        return results

    def save_model(self, filepath):
        if self.best_model is not None:
            joblib.dump({'model': self.best_model, 'feature_extractor': self.feature_extractor}, filepath)
            print(f' Modelo salvo: {filepath}')

    def load_model(self, filepath):
        loaded = joblib.load(filepath)
        self.best_model = loaded['model']
        self.feature_extractor = loaded['feature_extractor']
        return self

    def predict(self, features: Dict[str, float]) -> float:
        if self.best_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        prediction = self.best_model.predict(feature_vector)[0]
        return max(0, prediction)  # Ensure non-negative
