
import os
from typing import Dict, Any

class Config:
    

    def __init__(self):
        self.database = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'boostsql'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '123')
        }

        self.data_dir = os.getenv('DATA_DIR', 'data')
        self.models_dir = os.getenv('MODELS_DIR', 'models')

        self.query_generation = {
            'total_queries': int(os.getenv('TOTAL_QUERIES', '1250')),
            'complexities': ['simple', 'medium', 'complex'],
            'validation_timeout': int(os.getenv('VALIDATION_TIMEOUT', '30'))
        }

        self.ml_training = {
            'test_size': float(os.getenv('ML_TEST_SIZE', '0.2')),
            'random_state': int(os.getenv('ML_RANDOM_STATE', '42')),
            'cv_folds': int(os.getenv('ML_CV_FOLDS', '5')),
            'models': {
                'random_forest': {
                    'n_estimators': int(os.getenv('RF_N_ESTIMATORS', '100')),
                    'max_depth': int(os.getenv('RF_MAX_DEPTH', '10')) if os.getenv('RF_MAX_DEPTH') else None,
                    'random_state': int(os.getenv('ML_RANDOM_STATE', '42'))
                },
                'gradient_boosting': {
                    'n_estimators': int(os.getenv('GB_N_ESTIMATORS', '100')),
                    'learning_rate': float(os.getenv('GB_LEARNING_RATE', '0.1')),
                    'max_depth': int(os.getenv('GB_MAX_DEPTH', '3')),
                    'random_state': int(os.getenv('ML_RANDOM_STATE', '42'))
                },
                'linear_regression': {},
                'xgboost': {
                    'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', '100')),
                    'learning_rate': float(os.getenv('XGB_LEARNING_RATE', '0.1')),
                    'max_depth': int(os.getenv('XGB_MAX_DEPTH', '3')),
                    'random_state': int(os.getenv('ML_RANDOM_STATE', '42'))
                },
                'lightgbm': {
                    'n_estimators': int(os.getenv('LGB_N_ESTIMATORS', '100')),
                    'learning_rate': float(os.getenv('LGB_LEARNING_RATE', '0.1')),
                    'max_depth': int(os.getenv('LGB_MAX_DEPTH', '3')),
                    'random_state': int(os.getenv('ML_RANDOM_STATE', '42'))
                }
            }
        }

        self.feature_extraction = {
            'use_sqlparse': os.getenv('USE_SQLPARSE', 'false').lower() == 'true',
            'fallback_features': os.getenv('FALLBACK_FEATURES', 'true').lower() == 'true'
        }

        self.execution = {
            'batch_size': int(os.getenv('EXECUTION_BATCH_SIZE', '100')),
            'iterations': int(os.getenv('EXECUTION_ITERATIONS', '2')),
            'timeout': int(os.getenv('EXECUTION_TIMEOUT', '300'))
        }

        self.feature_extraction = {
            'use_sqlparse': os.getenv('USE_SQLPARSE', 'false').lower() == 'true',
            'fallback_features': os.getenv('FALLBACK_FEATURES', 'true').lower() == 'true'
        }

        self.llm = {
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'model': os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            'temperature': float(os.getenv('LLM_TEMPERATURE', '0.1')),
            'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '500'))
        }

    def get_database_config(self) -> Dict[str, Any]:
        
        return self.database.copy()

    def get_query_file_path(self) -> str:
        
        return os.path.join(self.data_dir, f"queries_{self.query_generation['total_queries']}.json")

    def get_execution_file_path(self) -> str:
        
        return os.path.join(self.data_dir, "queries_with_execution_times.json")

    def get_model_file_path(self, model_name: str = "trained_ml_agent") -> str:
        
        return os.path.join(self.models_dir, f"{model_name}.pkl")

    def get_training_summary_path(self) -> str:
        
        return os.path.join(self.models_dir, "training_summary.json")

config = Config()