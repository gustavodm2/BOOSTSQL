import sys
import os
import json
import random
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryInfo:
    
    query_id: int
    query_sql: str
    complexity: str
    estimated_complexity_score: int

class RobustQueryGenerator:
    

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.db_connector: Optional[Any] = None
        self.tables: Dict[str, List[str]] = {}
        self.table_relationships: Dict[str, List[Tuple[str, str, str]]] = {}
        self.column_types: Dict[str, Dict[str, str]] = {}
        self.table_row_counts: Dict[str, int] = {}
        self.sample_values: Dict[str, Dict[str, List[Any]]] = {}

        self._initialize_known_schema()
        self._connect_and_validate()

    def _initialize_known_schema(self):
        
        self.known_relationships = {
            'products': [('categories', 'category_id', 'id'), ('suppliers', 'supplier_id', 'id')],
            'orders': [('users', 'user_id', 'id'), ('customers', 'customer_id', 'id')],
            'order_items': [('orders', 'order_id', 'id'), ('products', 'product_id', 'id')],
            'payments': [('orders', 'order_id', 'id')],
            'user_sessions': [('users', 'user_id', 'id')],
            'product_reviews': [('products', 'product_id', 'id'), ('users', 'user_id', 'id')],
            'employees': [('departments', 'department_id', 'id')],
            'inventory': [('products', 'product_id', 'id')],
            'sales': [('products', 'product_id', 'id'), ('employees', 'employee_id', 'id')],
            'transactions': [('users', 'user_id', 'id')],
            'logs': [('users', 'user_id', 'id')],
            'profiles': [('users', 'user_id', 'id')],
            'shipping': [('orders', 'order_id', 'id')]
        }

    def _connect_and_validate(self):
        
        try:
            from src.database_connector import DatabaseConnector
            self.db_connector = DatabaseConnector(self.db_config)
            logger.info(' Connected to database')

            schema = self.db_connector.discover_schema()
            if not schema:
                raise Exception("No tables found in database")

            self._analyze_database_schema(schema)
            self._collect_table_statistics()
            self._collect_sample_data()

            logger.info(f' Database ready: {len(self.tables)} tables, {sum(self.table_row_counts.values())} total rows')

        except Exception as e:
            logger.error(f' Database connection/validation failed: {e}')
            raise

    def _analyze_database_schema(self, schema):
        
        for table, columns in schema.items():
            if table in self.known_relationships:
                self.tables[table] = columns
                self.table_relationships[table] = self.known_relationships[table]

        for table in schema:
            if table not in self.tables:
                self.tables[table] = schema[table]
                self.table_relationships[table] = []

        self._get_column_types()

    def _get_column_types(self):
        
        assert self.db_connector is not None
        try:
            with self.db_connector.get_connection() as conn:
                cursor = conn.cursor()
                for table in self.tables:
                    self.column_types[table] = {}
                    cursor.execute(, (table,))

                    for row in cursor.fetchall():
                        col_name, data_type = row
                        self.column_types[table][col_name] = data_type
        except Exception as e:
            logger.warning(f'Could not get column types: {e}')

    def _collect_table_statistics(self):
        
        assert self.db_connector is not None
        for table in self.tables:
            try:
                with self.db_connector.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    self.table_row_counts[table] = count
            except Exception as e:
                logger.warning(f'Could not get row count for {table}: {e}')
                self.table_row_counts[table] = 0

    def _collect_sample_data(self):
        
        assert self.db_connector is not None
        for table in self.tables:
            self.sample_values[table] = {}
            if self.table_row_counts[table] == 0:
                continue

            try:
                with self.db_connector.get_connection() as conn:
                    cursor = conn.cursor()

                    sample_size = min(50, self.table_row_counts[table])
                    cursor.execute(f"SELECT * FROM {table} TABLESAMPLE BERNOULLI(5) LIMIT {sample_size}")
                    rows = cursor.fetchall()

                    for col_idx, col_name in enumerate(self.tables[table]):
                        values = [row[col_idx] for row in rows if row[col_idx] is not None]
                        if values:
                            unique_values = list(set(values))[:10]
                            self.sample_values[table][col_name] = unique_values

            except Exception as e:
                logger.debug(f'Sample data collection failed for {table}: {e}')

    def _validate_query_safely(self, query: str) -> bool:
        
        assert self.db_connector is not None
        try:
            with self.db_connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN {query}")
                return True
        except Exception as e:
            logger.debug(f'Query validation failed: {e}')
            return False

    def generate_order_analytics_query(self) -> Optional[str]:
        
        if 'orders' not in self.tables or 'order_items' not in self.tables:
            return None

        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_product_performance_query(self) -> Optional[str]:
        
        if 'products' not in self.tables:
            return None

        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_user_behavior_query(self) -> Optional[str]:
        
        if 'users' not in self.tables:
            return None

        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_inventory_management_query(self) -> Optional[str]:
        
        if 'inventory' not in self.tables or 'products' not in self.tables:
            return None

        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_employee_department_query(self) -> Optional[str]:
        
        if 'employees' not in self.tables or 'departments' not in self.tables:
            return None

        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_time_series_revenue_query(self) -> Optional[str]:
        
        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_customer_segmentation_query(self) -> Optional[str]:
        
        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_supply_chain_query(self) -> Optional[str]:
        
        if 'suppliers' not in self.tables or 'products' not in self.tables:
            return None

        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_advanced_analytics_query(self) -> Optional[str]:
        
        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_comprehensive_business_query(self) -> Optional[str]:
        
        query = 

        return query.strip() if self._validate_query_safely(query) else None

    def generate_queries(self, total_queries: int = 100000) -> List[Dict]:
        
        query_generators = [
            self.generate_order_analytics_query,
            self.generate_product_performance_query,
            self.generate_user_behavior_query,
            self.generate_inventory_management_query,
            self.generate_employee_department_query,
            self.generate_time_series_revenue_query,
            self.generate_customer_segmentation_query,
            self.generate_supply_chain_query,
            self.generate_advanced_analytics_query,
            self.generate_comprehensive_business_query,
        ]

        queries = []
        generator_index = 0
        attempts = 0
        max_attempts = total_queries * 5

        logger.info(f" Generating {total_queries} REAL complex queries...")

        while len(queries) < total_queries and attempts < max_attempts:
            attempts += 1

            try:
                generator = query_generators[generator_index % len(query_generators)]
                query = generator()

                if query:
                    queries.append({
                        'query_id': len(queries) + 1,
                        'query_sql': query,
                        'complexity': 'complex',
                        'estimated_complexity_score': len(query.split()) + (query.count('JOIN') * 10) + (query.count('WITH') * 15) + (query.count('OVER') * 8) + (query.count('CASE') * 5)
                    })

                    if len(queries) % 1000 == 0:
                        logger.info(f" Generated {len(queries)}/{total_queries} complex queries")

                generator_index += 1

            except Exception as e:
                logger.debug(f'Query generation failed: {e}')
                generator_index += 1
                continue

        logger.info(f" Successfully generated {len(queries)} complex queries")
        return queries

def main():
    logger.info(" SQLBoost Robust Query Generator Starting...")

    db_config = {'host': 'localhost', 'port': '5432', 'database': 'boostsql', 'user': 'postgres', 'password': '123'}

    try:
        generator = RobustQueryGenerator(db_config)
        queries = generator.generate_queries(1000)

        os.makedirs('data', exist_ok=True)
        filepath = 'data/complex_queries_1k.json'
        with open(filepath, 'w') as f:
            json.dump(queries, f, indent=2)

        logger.info(f" Complex queries saved to {filepath}")
        logger.info(f" File exists: {os.path.exists(filepath)}")

        total_queries = len(queries)
        if total_queries > 0:
            avg_complexity = sum(q['estimated_complexity_score'] for q in queries) / total_queries
            logger.info(" Query Generation Summary:")
            logger.info(f"  Total queries: {total_queries}")
            logger.info(f"  Average complexity score: {avg_complexity:.1f}")
            logger.info("  All queries are highly complex with advanced SQL features")

        logger.info(" Robust query generation completed!")

    except Exception as e:
        logger.error(f" Query generation failed: {e}")
        raise

if __name__ == '__main__':
    main()