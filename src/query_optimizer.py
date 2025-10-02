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
        else:
            self.model = None
    
    def prepare_training_data(self, queries_data):
        """Prepara dados de treinamento a partir de queries e tempos de execução"""
        features = []
        execution_times = []
        
        print("Extraindo características das queries...")
        for i, query_info in enumerate(queries_data):
            query = query_info['query_sql']
            exec_time = query_info['execution_time_ms']
            
            # Extrai características
            query_features = self.feature_extractor.extract_features(query)
            features.append(list(query_features.values()))
            execution_times.append(exec_time)
            
            if (i + 1) % 100 == 0:
                print(f"Processadas {i + 1} queries...")
        
        X = pd.DataFrame(features, columns=self.feature_extractor.feature_names)
        y = np.array(execution_times)
        
        print(f"✅ Dataset preparado: {X.shape[0]} amostras, {X.shape[1]} características")
        return X, y
    
    def train(self, queries_data, save_path='models/sqlboost_model.pkl'):
        """Treina o modelo de otimização"""
        X, y = self.prepare_training_data(queries_data)
        self.model_trainer.feature_extractor = self.feature_extractor
        
        results = self.model_trainer.train_models(X, y)
        self.model_trainer.save_model(save_path)
        
        return results
    
    def predict_execution_time(self, query):
        """Prediz o tempo de execução de uma query"""
        if self.model_trainer.best_model is None:
            raise ValueError("Modelo não treinado. Execute o treinamento primeiro.")
        
        features = self.feature_extractor.extract_features(query)
        X = pd.DataFrame([list(features.values())], 
                        columns=self.feature_extractor.feature_names)
        
        return self.model_trainer.best_model.predict(X)[0]
    
    def suggest_optimizations(self, query):
        """Sugere otimizações baseadas nas características da query"""
        features = self.feature_extractor.extract_features(query)
        suggestions = []
        
        # Heurísticas inteligentes baseadas nas características
        if features['num_joins'] > 3:
            suggestions.append("🔍 Muitos JOINs - Considere usar subqueries ou materialized views")
        
        if features['num_subqueries'] > 2:
            suggestions.append("🔄 Muitas subqueries - Avalie usar CTEs (Common Table Expressions)")
        
        if features['has_aggregation'] and not features['has_group_by']:
            suggestions.append("📊 Agregação sem GROUP BY - Verifique se é intencional")
        
        if features['nested_level'] > 3:
            suggestions.append("🌀 Query muito aninhada - Considere simplificar a lógica")
        
        if features['query_length'] > 1000:
            suggestions.append("📏 Query muito longa - Divida em queries menores ou use stored procedures")
        
        if features['num_conditions'] > 5:
            suggestions.append("🎯 Muitas condições - Otimize os índices das colunas usadas no WHERE")
        
        predicted_time = self.predict_execution_time(query)
        complexity_score = self._calculate_complexity_score(features)
        
        return {
            'predicted_execution_time_ms': predicted_time,
            'suggestions': suggestions,
            'query_complexity': complexity_score,
            'features': features
        }
    
    def _calculate_complexity_score(self, features):
        """Calcula um score de complexidade da query (0-1)"""
        weights = {
            'num_joins': 0.2,
            'num_subqueries': 0.15,
            'num_conditions': 0.15,
            'nested_level': 0.1,
            'join_complexity': 0.15,
            'query_length': 0.1,
            'num_tables': 0.15
        }
        
        score = 0
        for key, weight in weights.items():
            if key in features:
                # Normaliza cada feature para 0-1
                normalized_value = min(features[key] / 10.0, 1.0)
                score += normalized_value * weight
        
        return min(score, 1.0)