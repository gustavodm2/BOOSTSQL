import sys
import os
import json
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_lots_queries():
    logger.info('🚀 GENERATING LOTS OF QUERIES FOR YOUR DATABASE...')
    db_config = {'host': 'localhost', 'port': '5432', 'database': 'boostsql', 'user': 'postgres', 'password': '123'}
    try:
        from src.database_connector import DatabaseConnector
        from src.query_generator import MassiveQueryGenerator

        logger.info('🔌 Connecting to your PostgreSQL database...')
        db_connector = DatabaseConnector(db_config)
        logger.info('✅ Database connection successful!')

        logger.info('🔍 Discovering your database schema...')
        schema = db_connector.discover_schema()
        if not schema:
            logger.error('❌ No tables found in database!')
            return
        logger.info(f'📊 Found {len(schema)} tables: {list(schema.keys())}')

        query_gen = MassiveQueryGenerator()
        query_gen.adapt_to_database(db_connector)

        logger.info('⚡ Generating 50,000 queries of different types...')
        queries = []
        complexities = ['simple', 'medium', 'complex', 'expert']
        queries_per_complexity = 50000 // len(complexities)  # 12,500 each

        for complexity in complexities:
            logger.info(f'🎯 Generating {queries_per_complexity} {complexity} queries...')
            for i in range(queries_per_complexity):
                try:
                    if complexity == 'simple':
                        query = query_gen._generate_simple_query()
                    elif complexity == 'medium':
                        query = query_gen._generate_medium_query()
                    elif complexity == 'complex':
                        query = query_gen._generate_complex_query()
                    else:
                        query = query_gen._generate_expert_query()

                    # Validate query
                    if db_connector.validate_query(query):
                        queries.append({'query': query, 'complexity': complexity})
                    else:
                        logger.debug(f'Invalid query skipped: {query[:50]}...')
                except Exception as e:
                    logger.debug(f'Query generation failed: {e}')
                    continue

        logger.info(f'✅ Generated {len(queries)} valid queries')

        # Save to file
        os.makedirs('data', exist_ok=True)
        with open('data/lots_queries.json', 'w') as f:
            json.dump(queries, f, indent=2)

        logger.info('💾 Queries saved to data/lots_queries.json')
        logger.info('🎉 Query generation completed!')

    except ImportError as e:
        logger.error(f'❌ Missing dependencies: {e}')
        logger.error('Install required packages: pip install -r requirements.txt')
    except Exception as e:
        logger.error(f'❌ Query generation failed: {e}')

if __name__ == '__main__':
    generate_lots_queries()