#!/usr/bin/env python3
"""
RUN REAL ML AGENT - Execute this to train ML models on your actual PostgreSQL database!

This script:
1. Uses the query generator to create SQL queries
2. Executes them on your real PostgreSQL database
3. Measures actual execution times
4. Trains ML models on real performance data
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_real_ml_agent():
    """Run the complete real ML agent pipeline"""

    logger.info("🚀 BOOSTSQL REAL ML AGENT STARTING...")
    logger.info("This will train ML models using REAL database performance data!")

    # YOUR DATABASE CONFIGURATION - CHANGE THESE VALUES!
    db_config = {
        'host': 'localhost',           # Your PostgreSQL host
        'port': '5432',               # Your PostgreSQL port
        'database': 'boostsql',   # CHANGE THIS: Your database name
        'user': 'postgres',      # CHANGE THIS: Your PostgreSQL username
        'password': '123'   # CHANGE THIS: Your PostgreSQL password
    }

    try:
        # Import here to avoid import errors if packages not installed
        from src.database_connector import DatabaseConnector
        from src.query_generator import MassiveQueryGenerator
        from src.feature_extractor import SQLFeatureExtractor
        from src.model_trainer import ModelTrainer
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        logger.info("🔌 Connecting to your PostgreSQL database...")

        # 1. Connect to real database
        db_connector = DatabaseConnector(db_config)
        logger.info("✅ Database connection successful!")

        # 2. Discover schema
        logger.info("🔍 Discovering your database schema...")
        schema = db_connector.discover_schema()

        if not schema:
            logger.error("❌ No tables found in database!")
            logger.error("Make sure your database has tables and you have permission to access them.")
            return

        logger.info(f"📊 Found {len(schema)} tables: {list(schema.keys())}")

        # 3. Initialize components
        query_gen = MassiveQueryGenerator()
        feature_extractor = SQLFeatureExtractor()
        model_trainer = ModelTrainer()

        # 4. Adapt query generator to real schema
        logger.info("🔄 Adapting query generator to your database...")
        query_gen.tables = list(schema.keys())
        query_gen.columns = schema

        # 5. Generate and execute queries to collect real performance data
        logger.info("⚡ Generating queries and measuring REAL execution times...")

        performance_data = []
        sample_size = 200  # Adjust based on your database size

        complexities = ['simple', 'medium', 'complex']
        queries_per_complexity = sample_size // len(complexities)

        for complexity in complexities:
            logger.info(f"🎯 Generating {queries_per_complexity} {complexity} queries...")

            successful = 0
            for i in tqdm(range(queries_per_complexity), desc=f"{complexity}"):
                try:
                    # Generate query
                    if complexity == 'simple':
                        query = query_gen._generate_simple_query()
                    elif complexity == 'medium':
                        query = query_gen._generate_medium_query()
                    else:
                        query = query_gen._generate_complex_query()

                    # Validate and execute
                    if db_connector.validate_query(query):
                        result = db_connector.execute_query_with_timing(query, iterations=2)

                        if result['success']:
                            features = feature_extractor.extract_features(query)

                            performance_data.append({
                                'query': query,
                                'execution_time_ms': result['execution_time_ms'],
                                'features': features,
                                'complexity': complexity
                            })

                            successful += 1

                except Exception as e:
                    continue

            logger.info(f"✅ Collected {successful} successful {complexity} query measurements")

        # 6. Train ML model on real data
        if len(performance_data) < 50:
            logger.error(f"❌ Only collected {len(performance_data)} measurements. Need at least 50 for training.")
            return

        logger.info(f"🧠 Training ML model on {len(performance_data)} real performance measurements...")

        # Prepare training data
        features_list = [list(item['features'].values()) for item in performance_data]
        execution_times = [item['execution_time_ms'] for item in performance_data]

        X = pd.DataFrame(features_list, columns=feature_extractor.feature_names)
        y = np.array(execution_times)

        # Train models
        results = model_trainer.train_models(X, y)

        # Save the best model
        os.makedirs('models', exist_ok=True)
        model_trainer.save_model('models/real_ml_agent.pkl')

        # 7. Show results
        logger.info("🎉 REAL ML TRAINING COMPLETED!")
        logger.info(f"🤖 Best model: {max(results.keys(), key=lambda x: results[x]['r2'])}")
        logger.info(f"📊 Training data: {len(performance_data)} real query executions")
        logger.info(f"⏱️ Execution time range: {min(execution_times):.2f}ms - {max(execution_times):.2f}ms")

        # Show sample predictions
        logger.info("\n🔮 Testing model predictions:")
        test_queries = performance_data[:3]
        for item in test_queries:
            features = pd.DataFrame([list(item['features'].values())],
                                  columns=feature_extractor.feature_names)
            predicted = model_trainer.best_model.predict(features)[0]
            actual = item['execution_time_ms']
            logger.info(f"  Query: {item['query'][:50]}...")
            logger.info(f"  Predicted: {predicted:.2f}ms | Actual: {actual:.2f}ms | Error: {abs(predicted-actual):.2f}ms")

        logger.info("\n💾 Model saved as: models/real_ml_agent.pkl")
        logger.info("🚀 You now have a REAL ML-powered SQL optimizer!")

    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.error("Install required packages: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"❌ Real ML Agent failed: {e}")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("1. Update db_config with your real PostgreSQL credentials")
        logger.info("2. Ensure PostgreSQL is running and accessible")
        logger.info("3. Make sure your database has tables with data")
        logger.info("4. Check that you have SELECT permissions on the tables")

if __name__ == "__main__":
    run_real_ml_agent()