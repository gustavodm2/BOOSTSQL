import sys
import os
import json
import time
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_queries():
    logger.info('🚀 EXECUTING QUERIES ON DATABASE...')

    # Database config
    db_config = {'host': 'localhost', 'port': '5432', 'database': 'boostsql', 'user': 'postgres', 'password': '123'}

    try:
        from src.database_connector import DatabaseConnector

        logger.info('🔌 Connecting to PostgreSQL database...')
        db_connector = DatabaseConnector(db_config)
        logger.info('✅ Database connection successful!')

        # Load queries
        logger.info('📂 Loading queries from data/queries_1250.json...')
        with open('data/queries_1250.json', 'r') as f:
            queries = json.load(f)

        logger.info(f'📊 Loaded {len(queries)} queries')

        # Execute queries in batches to avoid memory issues
        batch_size = 1000
        updated_queries = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            logger.info(f'⚡ Executing batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size} ({len(batch)} queries)...')

            for query_data in batch:
                try:
                    # Execute query directly and measure time
                    start_time = time.time()

                    with db_connector.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(query_data['query_sql'])
                        if cursor.description:
                            results = cursor.fetchall()
                            result_count = len(results)
                        else:
                            result_count = 0
                        conn.commit()

                    end_time = time.time()
                    actual_time = (end_time - start_time) * 1000  # Convert to milliseconds

                    # Update query data with actual execution time
                    updated_query = query_data.copy()
                    updated_query['actual_execution_time_ms'] = actual_time
                    updated_query['execution_success'] = True
                    updated_query['result_count'] = result_count
                    updated_query['error'] = None

                    updated_queries.append(updated_query)

                    # Log progress every 100 queries
                    if len(updated_queries) % 100 == 0:
                        logger.info(f'  ✅ Executed {len(updated_queries)}/{len(queries)} queries')

                except Exception as e:
                    logger.warning(f'❌ Query {query_data["query_id"]} failed: {e}')
                    # Still add the query with failure info
                    updated_query = query_data.copy()
                    updated_query['actual_execution_time_ms'] = None
                    updated_query['execution_success'] = False
                    updated_query['result_count'] = 0
                    updated_query['error'] = str(e)
                    updated_queries.append(updated_query)

        # Save results
        logger.info('💾 Saving execution results...')
        os.makedirs('data', exist_ok=True)
        with open('data/queries_with_execution_times.json', 'w') as f:
            json.dump(updated_queries, f, indent=2)

        logger.info('✅ Query execution completed!')

        # Print summary
        successful = sum(1 for q in updated_queries if q.get('execution_success', False))
        failed = len(updated_queries) - successful

        logger.info('📊 Execution Summary:')
        logger.info(f'  ✅ Successful: {successful} queries')
        logger.info(f'  ❌ Failed: {failed} queries')
        logger.info(f'  📁 Results saved to: data/queries_with_execution_times.json')

        # Show some timing stats
        successful_times = [q['actual_execution_time_ms'] for q in updated_queries if q.get('execution_success') and q.get('actual_execution_time_ms')]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            min_time = min(successful_times)
            max_time = max(successful_times)
            logger.info('⏱️  Timing Statistics (successful queries):')
            logger.info(f'  📈 Average: {avg_time:.2f}ms')
            logger.info(f'  🏃 Min: {min_time:.2f}ms')
            logger.info(f'  🐌 Max: {max_time:.2f}ms')

    except ImportError as e:
        logger.error(f'❌ Missing dependencies: {e}')
        logger.error('Install required packages: pip install -r requirements.txt')
    except FileNotFoundError:
        logger.error('❌ Query file not found: data/queries_1250.json')
        logger.error('Run python scripts/generate_queries.py first')
    except Exception as e:
        logger.error(f'❌ Query execution failed: {e}')

if __name__ == '__main__':
    execute_queries()