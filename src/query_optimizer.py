import pandas as pd
import numpy as np
from .feature_extractor import SQLFeatureExtractor
from .model_trainer import ModelTrainer

class SQLBoostOptimizer:
    def __init__(self, model_path=None):
        self.feature_extractor = SQLFeatureExtractor()
        self.model_trainer = ModelTrainer()
        
        if model_path:
            self.model_trainer.load_model(model_path)
    
    def prepare_training_data(self, queries_data):
        """Prepara dados de treinamento para 500k queries"""
        print("🔨 Extraindo características...")
        
        features = []
        execution_times = []
        
        for i, query_info in enumerate(queries_data):
            if i % 50000 == 0:
                print(f"📊 Processadas {i}/{len(queries_data)} queries...")
            
            query = query_info['query_sql']
            exec_time = query_info['execution_time_ms']
            
            query_features = self.feature_extractor.extract_features(query)
            features.append(list(query_features.values()))
            execution_times.append(exec_time)
        
        X = pd.DataFrame(features, columns=self.feature_extractor.feature_names)
        y = np.array(execution_times)
        
        print(f"✅ Dataset preparado: {X.shape[0]} amostras")
        return X, y
    
    def train(self, queries_data, save_path='models/sqlboost_500k.pkl'):
        """Treina modelo com 500k queries"""
        X, y = self.prepare_training_data(queries_data)
        self.model_trainer.feature_extractor = self.feature_extractor
        
        results = self.model_trainer.train_models(X, y)
        self.model_trainer.save_model(save_path)
        
        return results
    
    def predict_execution_time(self, query):
        """Prediz tempo de execução"""
        if self.model_trainer.best_model is None:
            raise ValueError("Modelo não treinado!")
        
        features = self.feature_extractor.extract_features(query)
        X = pd.DataFrame([list(features.values())], 
                        columns=self.feature_extractor.feature_names)
        
        return self.model_trainer.best_model.predict(X)[0]
    
    def suggest_optimizations(self, query):
        """Sugere otimizações inteligentes"""
        features = self.feature_extractor.extract_features(query)
        suggestions = []
        
        # Heurísticas baseadas em características
        if features['num_joins'] > 3:
            suggestions.append("🔍 Múltiplos JOINs - Considere usar CTEs")
        
        if features['num_subqueries'] > 2:
            suggestions.append("🔄 Muitas subqueries - Avalie usar JOINs")
        
        if features['nested_level'] > 3:
            suggestions.append("🌀 Query muito aninhada - Simplifique")
        
        if features['has_window_functions'] and features['num_tables'] > 2:
            suggestions.append("📊 Window functions complexas - Verifique performance")
        
        predicted_time = self.predict_execution_time(query)
        
        return {
            'predicted_execution_time_ms': predicted_time,
            'suggestions': suggestions,
            'query_complexity': self._calculate_complexity(features),
            'features_analysis': self._analyze_features(features)
        }
    
    def _calculate_complexity(self, features):
        """Calcula complexidade da query"""
        weights = {
            'num_joins': 0.15, 'num_subqueries': 0.12, 'num_conditions': 0.10,
            'nested_level': 0.08, 'query_length': 0.07, 'num_tables': 0.10,
            'has_window_functions': 0.08, 'join_complexity': 0.12
        }
        
        score = 0
        for key, weight in weights.items():
            if key in features:
                normalized = min(features[key] / 10.0, 1.0)
                score += normalized * weight
        
        return min(score, 1.0)
    
    def _analyze_features(self, features):
        """Analisa características da query"""
        analysis = {}
        
        if features['num_joins'] > 3:
            analysis['join_complexity'] = 'ALTA'
        elif features['num_joins'] > 1:
            analysis['join_complexity'] = 'MÉDIA'
        else:
            analysis['join_complexity'] = 'BAIXA'
        
        if features['num_subqueries'] > 2:
            analysis['subquery_usage'] = 'EXCESSIVO'
        elif features['num_subqueries'] > 0:
            analysis['subquery_usage'] = 'MODERADO'
        else:
            analysis['subquery_usage'] = 'NENHUM'
        
        return analysis