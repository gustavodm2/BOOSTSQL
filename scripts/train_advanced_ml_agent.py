import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_advanced_ml_agent():
    logger.info(' SQLBoost Advanced ML Agent Training Starting...')
    logger.info('This agent learns from execution data and develops optimization strategies!')

    try:
        from config import config
        from src.ml_agent import SQLBoostMLAgent

        db_config = config.get_database_config()

        agent = SQLBoostMLAgent(db_config)

        execution_file = config.get_execution_file_path()
        logger.info(f' Loading execution results from {execution_file}...')

        if not os.path.exists(execution_file):
            logger.error(f' Execution results file not found: {execution_file}')
            logger.error('Run python scripts/execute_queries.py first to collect performance data')
            return

        logger.info(' Training advanced ML agent...')
        agent.learn_from_execution_data(execution_file)

        logger.info(' Running adaptive learning...')
        agent.adaptive_learning()

        status = agent.get_agent_status()

        logger.info(' ADVANCED ML AGENT TRAINING COMPLETED!')
        logger.info(' Agent Status:')
        logger.info(f'  Knowledge base patterns: {status["knowledge_base_patterns"]}')
        logger.info(f'  Optimizations learned: {status["optimizations_learned"]}')
        logger.info(f'  Strategies available: {status["strategies_available"]}')
        logger.info(f'  Exploration rate: {status["exploration_rate"]:.3f}')
        logger.info(f'  Learning rate: {status["learning_rate"]:.3f}')

        logger.info('\n Strategy Effectiveness:')
        for strategy_name, metrics in status['strategy_effectiveness'].items():
            logger.info(f'  {strategy_name}:')
            logger.info(f'    Success rate: {metrics["success_rate"]:.2%}')
            logger.info(f'    Avg improvement: {metrics["avg_improvement"]:.2f}x')
            logger.info(f'    Attempts: {metrics["attempts"]}')

        logger.info('\n Agent knowledge saved to: models/agent_knowledge.json')
        logger.info(' You now have an advanced ML-powered SQL optimizer!')
        logger.info('\n Next steps:')
        logger.info('  1. Use the agent with: python scripts/use_advanced_ml_agent.py')
        logger.info('  2. Generate more queries for better learning: python scripts/generate_queries.py')
        logger.info('  3. Execute more queries to collect more data: python scripts/execute_queries.py')

    except ImportError as e:
        logger.error(f' Missing dependencies: {e}')
        logger.error('Install required packages: pip install -r requirements.txt')
    except Exception as e:
        logger.error(f' Advanced ML Agent training failed: {e}')
        logger.info('\n Troubleshooting:')
        logger.info('1. Make sure you have run execute_queries.py to collect performance data')
        logger.info('2. Check that data/queries_with_execution_times.json exists')
        logger.info('3. Ensure the execution data contains timing measurements')

if __name__ == '__main__':
    train_advanced_ml_agent()