import psycopg2
import time
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import pandas as pd
from sqlalchemy import create_engine, text
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnector:

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.engine = None
        self._connect()

    def _connect(self):
        try:
            connection_string = f'postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}'
            self.engine = create_engine(connection_string)
            logger.info('Database connection established')
        except Exception as e:
            logger.error(f'Failed to connect to database: {e}')
            raise

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = True
            yield conn
        except Exception as e:
            logger.error(f'Database connection error: {e}')
            raise
        finally:
            if conn:
                conn.close()

    def execute_query_with_timing(self, query: str, iterations: int=3) -> Dict:
        execution_times = []
        results = None
        error = None
        for i in range(iterations):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    explain_query = f'EXPLAIN (ANALYZE, BUFFERS, TIMING) {query}'
                    cursor.execute(explain_query)
                    start_time = time.time()
                    cursor.execute(query)
                    if cursor.description:
                        results = cursor.fetchall()
                    conn.commit()
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000
                    execution_times.append(execution_time)
                    logger.debug(f'Iteration {i + 1}: {execution_time:.2f}ms')
            except Exception as e:
                error = str(e)
                logger.warning(f'Query execution failed: {e}')
                break
        if not execution_times:
            return {'success': False, 'error': error, 'execution_time_ms': None, 'std_dev': None, 'iterations_completed': 0}
        avg_time = sum(execution_times) / len(execution_times)
        std_dev = pd.Series(execution_times).std() if len(execution_times) > 1 else 0
        return {'success': True, 'execution_time_ms': avg_time, 'std_dev': std_dev, 'iterations_completed': len(execution_times), 'min_time': min(execution_times), 'max_time': max(execution_times), 'results_count': len(results) if results else 0}

    def discover_schema(self) -> Dict[str, List[str]]:
        schema_info = {}
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("\n                    SELECT table_name\n                    FROM information_schema.tables\n                    WHERE table_schema = 'public'\n                    AND table_type = 'BASE TABLE'\n                    ORDER BY table_name\n                ")
                tables = [row[0] for row in cursor.fetchall()]
                for table in tables:
                    cursor.execute("\n                        SELECT column_name\n                        FROM information_schema.columns\n                        WHERE table_name = %s\n                        AND table_schema = 'public'\n                        ORDER BY ordinal_position\n                    ", (table,))
                    columns = [row[0] for row in cursor.fetchall()]
                    schema_info[table] = columns
                logger.info(f'Discovered {len(schema_info)} tables in database')
        except Exception as e:
            logger.error(f'Schema discovery failed: {e}')
            raise
        return schema_info

    def get_table_stats(self) -> Dict[str, Dict]:
        stats = {}
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('\n                    SELECT\n                        schemaname,\n                        tablename,\n                        n_tup_ins,\n                        n_tup_upd,\n                        n_tup_del,\n                        n_live_tup,\n                        n_dead_tup\n                    FROM pg_stat_user_tables\n                    ORDER BY tablename\n                ')
                for row in cursor.fetchall():
                    table_name = row[1]
                    stats[table_name] = {'rows': row[5], 'inserts': row[2], 'updates': row[3], 'deletes': row[4], 'dead_rows': row[6]}
        except Exception as e:
            logger.warning(f'Could not get table stats: {e}')
        return stats

    def validate_query(self, query: str) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'EXPLAIN {query}')
                return True
        except Exception as e:
            logger.debug(f'Query validation failed: {e}')
            return False
