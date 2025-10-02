import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, max_depth=6)
        }
        self.best_model = None
        self.feature_extractor = None
    
    def train_models(self, X, y):
        """Treina e compara diferentes modelos"""
        results = {}
        
        # Ajusta número de folds baseado no tamanho do dataset
        n_samples = len(X)
        n_splits = min(5, n_samples // 10)  # Máximo 5 folds, mínimo 10 amostras por fold
        n_splits = max(2, n_splits)  # Pelo menos 2 folds
        
        print(f"Treinando com {n_samples} amostras e {n_splits} folds de validação cruzada")
        
        for name, model in self.models.items():
            print(f"Treinando {name}...")
            
            try:
                # Validação cruzada adaptativa
                if n_splits > 1:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
                    cv_mae_mean = -cv_scores.mean()
                    cv_mae_std = cv_scores.std()
                else:
                    # Split simples se não há amostras suficientes para CV
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv_mae_mean = mean_absolute_error(y_test, y_pred)
                    cv_mae_std = 0
                
                # Treinamento final com todos os dados
                model.fit(X, y)
                y_pred = model.predict(X)
                
                results[name] = {
                    'model': model,
                    'mae': mean_absolute_error(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'r2': r2_score(y, y_pred),
                    'cv_mae_mean': cv_mae_mean,
                    'cv_mae_std': cv_mae_std
                }
                
            except Exception as e:
                print(f"Erro no treinamento de {name}: {e}")
                continue
        
        if not results:
            raise ValueError("Nenhum modelo foi treinado com sucesso")
        
        # Seleciona o melhor modelo baseado no R²
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        
        print("\n=== Resultados dos Modelos ===")
        for name, metrics in results.items():
            print(f"\n{name.upper():<15}")
            print(f"MAE: {metrics['mae']:.2f} ms")
            print(f"RMSE: {metrics['rmse']:.2f} ms")
            print(f"R²: {metrics['r2']:.3f}")
            if n_splits > 1:
                print(f"CV MAE: {metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f} ms")
        
        print(f"\n🎯 Melhor modelo: {best_model_name}")
        return results
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'feature_extractor': self.feature_extractor
            }, filepath)
            print(f"💾 Modelo salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega um modelo salvo"""
        loaded = joblib.load(filepath)
        self.best_model = loaded['model']
        self.feature_extractor = loaded['feature_extractor']
        return self