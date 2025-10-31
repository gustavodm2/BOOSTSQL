import sys
import os
import json
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ml_agent():
    logger.info('🚀 SQLBoost ML Agent Training Starting...')
    logger.info('This will train ML models using collected performance data!')

    try:
        from config import config
        from src.feature_extractor import SQLFeatureExtractor
        from src.model_trainer import ModelTrainer
        import pandas as pd
        import numpy as np

        # Load execution results
        execution_file = config.get_execution_file_path()
        logger.info(f'📂 Loading execution results from {execution_file}...')
        try:
            with open(execution_file, 'r') as f:
                execution_data = json.load(f)
            logger.info(f'✅ Loaded {len(execution_data)} query execution results')
        except FileNotFoundError:
            logger.error(f'❌ Execution results file not found: {execution_file}')
            logger.error('Run python scripts/execute_queries.py first to collect performance data')
            return

        # Filter successful executions with timing data
        successful_executions = [
            item for item in execution_data
            if item.get('execution_success') and item.get('actual_execution_time_ms') is not None
        ]

        if len(successful_executions) < 10:
            logger.error(f'❌ Only {len(successful_executions)} successful executions found. Need at least 10 for training.')
            logger.error('Make sure execute_queries.py ran successfully and collected timing data.')
            return

        logger.info(f'✅ Found {len(successful_executions)} successful query executions with timing data')

        # Extract features and prepare training data
        logger.info('🔍 Extracting features from queries...')
        feature_extractor = SQLFeatureExtractor()
        model_trainer = ModelTrainer()

        training_data = []
        for item in successful_executions:
            try:
                query = item['query_sql']
                execution_time = item['actual_execution_time_ms']

                # Extract features from the query
                features = feature_extractor.extract_features(query)

                training_data.append({
                    'query': query,
                    'execution_time_ms': execution_time,
                    'features': features,
                    'complexity': item.get('complexity', 'unknown'),
                    'query_id': item.get('query_id')
                })

            except Exception as e:
                logger.warning(f'Skipping query due to feature extraction error: {e}')
                continue

        if len(training_data) < 10:
            logger.error(f'❌ Only {len(training_data)} queries with valid features. Need at least 10 for training.')
            return

        logger.info(f'✅ Prepared {len(training_data)} training samples with features')

        # Prepare data for ML training
        logger.info('🧠 Training ML models...')
        features_list = [list(item['features'].values()) for item in training_data]
        execution_times = [item['execution_time_ms'] for item in training_data]

        X = pd.DataFrame(features_list, columns=feature_extractor.feature_names)
        y = np.array(execution_times)

        # Train models
        results = model_trainer.train_models(X, y)

        # Save the best model
        os.makedirs('models', exist_ok=True)
        model_trainer.save_model('models/trained_ml_agent.pkl')

        logger.info('🎉 ML TRAINING COMPLETED!')

        # Show training statistics
        logger.info(f'📊 Training data: {len(training_data)} real query executions')
        logger.info(f'⏱️ Execution time range: {min(execution_times):.2f}ms - {max(execution_times):.2f}ms')
        logger.info(f'📈 Average execution time: {sum(execution_times)/len(execution_times):.2f}ms')

        # Show model performance details
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_result = results[best_model_name]
        logger.info(f'🏆 Best Model Performance:')
        logger.info(f'  Model: {best_model_name}')
        logger.info(f'  R² Score: {best_result["r2"]:.4f}')
        logger.info(f'  RMSE: {best_result["rmse"]:.2f}ms')
        logger.info(f'  MAE: {best_result["mae"]:.2f}ms')

        # Show all model results
        logger.info('\n📋 All Model Results:')
        for model_name, result in results.items():
            logger.info(f'  {model_name}:')
            logger.info(f'    R²: {result["r2"]:.4f} | RMSE: {result["rmse"]:.2f}ms | MAE: {result["mae"]:.2f}ms')

        # Test predictions on a few samples
        logger.info('\n🔮 Testing model predictions on sample queries:')
        test_samples = training_data[:5] if len(training_data) >= 5 else training_data

        for i, item in enumerate(test_samples):
            features_df = pd.DataFrame([list(item['features'].values())], columns=feature_extractor.feature_names)
            predicted = model_trainer.best_model.predict(features_df)[0]
            actual = item['execution_time_ms']

            logger.info(f'  Test {i+1}:')
            logger.info(f'    Query: {item["query"][:60]}...'[:60])
            logger.info(f'    Predicted: {predicted:.2f}ms | Actual: {actual:.2f}ms | Error: {abs(predicted - actual):.2f}ms')

        logger.info('\n💾 Model saved as: models/trained_ml_agent.pkl')
        logger.info('🚀 You now have a trained ML-powered SQL optimizer!')
        logger.info('\n💡 Next steps:')
        logger.info('  1. Use the model with: python scripts/use_model.py')
        logger.info('  2. Generate more queries for better training: python scripts/generate_queries.py')
        logger.info('  3. Execute more queries to collect more data: python scripts/execute_queries.py')

        # Save training summary
        training_summary = {
            'training_date': pd.Timestamp.now().isoformat(),
            'total_samples': len(training_data),
            'execution_time_stats': {
                'min': float(min(execution_times)),
                'max': float(max(execution_times)),
                'avg': float(sum(execution_times)/len(execution_times))
            },
            'best_model': best_model_name,
            'model_performance': {
                'r2': float(best_result['r2']),
                'rmse': float(best_result['rmse']),
                'mae': float(best_result['mae'])
            },
            'all_models': {name: {
                'r2': float(result['r2']),
                'rmse': float(result['rmse']),
                'mae': float(result['mae'])
            } for name, result in results.items()}
        }

        with open('models/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)

        logger.info('📄 Training summary saved to: models/training_summary.json')

    except ImportError as e:
        logger.error(f'❌ Missing dependencies: {e}')
        logger.error('Install required packages: pip install -r requirements.txt')
    except Exception as e:
        logger.error(f'❌ ML Agent training failed: {e}')
        logger.info('\n🔧 Troubleshooting:')
        logger.info('1. Make sure you have run execute_queries.py to collect performance data')
        logger.info('2. Check that data/queries_with_execution_times.json exists')
        logger.info('3. Ensure the execution data contains timing measurements')

if __name__ == '__main__':
    train_ml_agent()