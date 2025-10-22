#!/usr/bin/env python3
"""
Real ML Training Script for BOOSTSQL
Trains models on actual database performance data instead of synthetic data
"""

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database_connector import DatabaseConnector
from src.feature_extractor import SQLFeatureExtractor
from src.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealMLTrainer:
    """Trains ML models on real database performance data"""

    def __init__(self, db_config):
        self.db_connector = DatabaseConnector(db_config)
        self.feature_extractor = SQLFeatureExtractor()
        self.model_trainer = ModelTrainer()

    def collect_real_performance_data(self, query_templates, sample_size=1000):
        """
        Collect real execution times from database

        Args:
            query_templates: List of SQL query templates to test
            sample_size: Number of queries to collect

        Returns:
            List of dicts with query, features, and real execution time
        """
        logger.info(f"Collecting {sample_size} real performance measurements...")

        performance_data = []
        successful_measurements = 0

        for i in tqdm(range(sample_size), desc="Measuring queries"):
            try:
                # Generate or select a query
                if i < len(query_templates):
                    query = query_templates[i]
                else:
                    # Generate variations of existing queries
                    base_query = np.random.choice(query_templates)
                    query = self._vary_query(base_query)

                # Measure actual execution time
                result = self.db_connector.execute_query_with_timing(query, iterations=3)

                if result['success']:
                    # Extract features
                    features = self.feature_extractor.extract_features(query)

                    performance_data.append({
                        'query': query,
                        'execution_time_ms': result['execution_time_ms'],
                        'features': features,
                        'std_dev': result['std_dev'],
                        'iterations': result['iterations_completed']
                    })

                    successful_measurements += 1

                    if successful_measurements % 100 == 0:
                        logger.info(f"Collected {successful_measurements} successful measurements")

                else:
                    logger.warning(f"Query failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.warning(f"Error measuring query {i}: {e}")
                continue

        logger.info(f"Successfully collected {successful_measurements} performance measurements")
        return performance_data

    def _vary_query(self, base_query):
        """Create variations of a base query for more diverse training data"""
        # Simple variations: change LIMIT values, add/remove WHERE conditions, etc.
        variations = [
            lambda q: q.replace('LIMIT 10', 'LIMIT 100'),
            lambda q: q.replace('LIMIT 100', 'LIMIT 1000'),
            lambda q: q.replace('ORDER BY id', 'ORDER BY name'),
            lambda q: q + ' LIMIT 50' if 'LIMIT' not in q else q,
        ]

        variation = np.random.choice(variations)
        try:
            return variation(base_query)
        except:
            return base_query

    def prepare_training_dataset(self, performance_data):
        """Prepare dataset for ML training"""
        logger.info("Preparing training dataset...")

        features_list = []
        execution_times = []

        for item in performance_data:
            features_list.append(list(item['features'].values()))
            execution_times.append(item['execution_time_ms'])

        X = pd.DataFrame(features_list, columns=self.feature_extractor.feature_names)
        y = np.array(execution_times)

        logger.info(f"Training dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Execution time range: {y.min():.2f}ms - {y.max():.2f}ms")

        return X, y

    def train_real_ml_model(self, performance_data, save_path='models/real_ml_boost.pkl'):
        """Train ML model on real performance data"""
        X, y = self.prepare_training_dataset(performance_data)

        logger.info("Training ML models on real performance data...")
        results = self.model_trainer.train_models(X, y)

        # Save the trained model
        self.model_trainer.save_model(save_path)
        logger.info(f"Real ML model saved to {save_path}")

        return results

    def validate_model(self, performance_data, test_size=0.2):
        """Validate model performance on held-out data"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X, y = self.prepare_training_dataset(performance_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Retrain on training set
        self.model_trainer.train_models(X_train, y_train)

        # Predict on test set
        y_pred = self.model_trainer.best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info("Model Validation Results:")
        logger.info(f"MAE: {mae:.2f}ms")
        logger.info(f"RMSE: {rmse:.2f}ms")
        logger.info(f"R²: {r2:.3f}")

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    """Main training function"""

    # Database configuration - UPDATE THESE WITH YOUR REAL DATABASE
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'your_database',
        'user': 'your_user',
        'password': 'your_password'
    }

    # Example query templates - REPLACE WITH YOUR REAL QUERIES
    query_templates = [
        "SELECT * FROM users LIMIT 10",
        "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
        "SELECT * FROM products WHERE price > 100",
        "SELECT u.name, p.name FROM users u JOIN orders o ON u.id = o.user_id JOIN products p ON o.product_id = p.id",
        "SELECT category, AVG(price) FROM products GROUP BY category",
        "SELECT * FROM users WHERE created_at > '2023-01-01'",
        "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 500)",
    ]

    try:
        # Initialize real ML trainer
        trainer = RealMLTrainer(db_config)

        # Collect real performance data
        logger.info("Starting real ML training pipeline...")
        performance_data = trainer.collect_real_performance_data(query_templates, sample_size=500)

        if len(performance_data) < 50:
            logger.error("Not enough successful measurements. Check database connection and queries.")
            return

        # Train model
        results = trainer.train_real_ml_model(performance_data)

        # Validate model
        validation_results = trainer.validate_model(performance_data)

        logger.info("Real ML training completed successfully!")
        logger.info(f"Collected {len(performance_data)} real performance measurements")
        logger.info(f"Best model: {max(results.keys(), key=lambda x: results[x]['r2'])}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Make sure to update db_config with your real database credentials")
        logger.info("Also ensure your database has the necessary tables and data")

if __name__ == "__main__":
    main()