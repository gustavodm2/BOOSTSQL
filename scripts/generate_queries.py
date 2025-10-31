import sys
import os
import json
import random
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleQueryGenerator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_connector = None
        self.tables = {}
        self.sample_values = {}
        self._connect_and_analyze()

    def _connect_and_analyze(self):
        """Connect to database and get basic info"""
        try:
            from src.database_connector import DatabaseConnector
            self.db_connector = DatabaseConnector(self.db_config)
            logger.info('🔌 Connected to database for query generation')

            # Get schema
            schema = self.db_connector.discover_schema()
            if not schema:
                raise Exception("No tables found")

            # Filter to tables with data
            for table, columns in schema.items():
                try:
                    with self.db_connector.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            self.tables[table] = columns
                            # Get some sample values
                            cursor.execute(f"SELECT * FROM {table} LIMIT 10")
                            rows = cursor.fetchall()
                            self.sample_values[table] = {}
                            for col_idx, col_name in enumerate(columns):
                                values = [row[col_idx] for row in rows if row[col_idx] is not None]
                                if values:
                                    self.sample_values[table][col_name] = list(set(values))  # unique values
                except Exception as e:
                    logger.warning(f'Skipping table {table}: {e}')

            logger.info(f'📊 Found {len(self.tables)} tables with data')

        except Exception as e:
            logger.error(f'❌ Database connection failed: {e}')
            raise

    def _get_sample_value(self, table, column):
        """Get a random sample value for a column"""
        if table in self.sample_values and column in self.sample_values[table]:
            values = self.sample_values[table][column]
            if values:
                return random.choice(values)
        return None

    def _validate_query(self, query):
        """Validate query can execute"""
        try:
            with self.db_connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN {query}")
                return True
        except Exception as e:
            logger.debug(f'Query validation failed: {e}')
            return False

    def generate_simple_query(self):
        """Generate simple validated queries"""
        table = random.choice(list(self.tables.keys()))
        columns = self.tables[table]

        # Simple SELECT
        if random.random() < 0.4:
            selected_cols = random.sample(columns, min(3, len(columns)))
            query = f"SELECT {', '.join(selected_cols)} FROM {table}"
            if random.random() < 0.3:
                query += f" LIMIT {random.randint(1, 50)}"
        # COUNT
        elif random.random() < 0.6:
            query = f"SELECT COUNT(*) FROM {table}"
        # Simple WHERE with real data
        else:
            column = random.choice(columns)
            value = self._get_sample_value(table, column)

            if value is not None:
                if isinstance(value, str):
                    # Escape quotes properly
                    escaped_value = value.replace("'", "''")
                    if random.random() < 0.7:
                        query = f"SELECT * FROM {table} WHERE {column} = '{escaped_value}'"
                    else:
                        query = f"SELECT * FROM {table} WHERE {column} LIKE '%{escaped_value[:3]}%'"
                elif isinstance(value, (int, float)):
                    operator = random.choice(['=', '>', '<'])
                    query = f"SELECT * FROM {table} WHERE {column} {operator} {value}"
                else:
                    query = f"SELECT * FROM {table} WHERE {column} = '{str(value)}'"
            else:
                # Fallback
                query = f"SELECT * FROM {table} LIMIT 10"

        return query if self._validate_query(query) else None

    def generate_medium_query(self):
        """Generate medium complexity queries"""
        table = random.choice(list(self.tables.keys()))

        # GROUP BY
        if random.random() < 0.4:
            group_cols = [col for col in self.tables[table] if col in ['category', 'city', 'country', 'status']]
            if group_cols:
                group_col = random.choice(group_cols)
                query = f"SELECT {group_col}, COUNT(*) FROM {table} GROUP BY {group_col}"
                return query if self._validate_query(query) else None

        # ORDER BY
        elif random.random() < 0.7:
            column = random.choice(self.tables[table])
            direction = random.choice(['ASC', 'DESC'])
            query = f"SELECT * FROM {table} ORDER BY {column} {direction} LIMIT {random.randint(5, 20)}"
            return query if self._validate_query(query) else None

        # Simple JOIN if possible
        else:
            # Look for potential joins
            join_candidates = []
            for t1 in self.tables:
                for t2 in self.tables:
                    if t1 != t2:
                        # Check if there's a common column name that might be a foreign key
                        common_cols = set(self.tables[t1]) & set(self.tables[t2])
                        if common_cols:
                            for col in common_cols:
                                if col.endswith('_id') or col in ['id']:
                                    join_candidates.append((t1, t2, col))

            if join_candidates:
                t1, t2, join_col = random.choice(join_candidates)
                query = f"SELECT {t1}.*, {t2}.name FROM {t1} JOIN {t2} ON {t1}.{join_col} = {t2}.id LIMIT 10"
                return query if self._validate_query(query) else None

        # Fallback to simple query
        return self.generate_simple_query()

    def generate_complex_query(self):
        """Generate complex queries - simplified version"""
        # For now, just return medium queries as complex ones
        # In a real implementation, this would include subqueries, window functions, etc.
        return self.generate_medium_query()

    def generate_queries(self, total_queries=500000):
        """Generate balanced validated queries"""
        queries_per_complexity = total_queries // 3
        queries = []

        logger.info(f"🎯 Generating {total_queries} validated queries ({queries_per_complexity} per complexity level)")

        complexities = ['simple', 'medium', 'complex']

        for complexity in complexities:
            logger.info(f"⚡ Generating {queries_per_complexity} {complexity} queries...")
            successful = 0
            attempts = 0
            max_attempts = queries_per_complexity * 5  # Allow more failures

            while successful < queries_per_complexity and attempts < max_attempts:
                attempts += 1

                try:
                    if complexity == 'simple':
                        query = self.generate_simple_query()
                    elif complexity == 'medium':
                        query = self.generate_medium_query()
                    else:
                        query = self.generate_complex_query()

                    if query:
                        queries.append({
                            'query_id': len(queries) + 1,
                            'query_sql': query,
                            'complexity': complexity,
                            'estimated_complexity_score': len(query.split()) + (query.count('JOIN') * 10) + (query.count('GROUP BY') * 5)
                        })
                        successful += 1

                        if successful % 1000 == 0:
                            logger.info(f"  ✅ Generated and validated {successful}/{queries_per_complexity} {complexity} queries")

                except Exception as e:
                    logger.debug(f'Failed to generate {complexity} query: {e}')
                    continue

            logger.info(f"✅ Successfully generated and validated {successful} {complexity} queries")

        # Shuffle queries
        random.shuffle(queries)

        logger.info(f"🎉 Total validated queries generated: {len(queries)}")
        return queries

def main():
    logger.info("🚀 SQLBoost Database-Aware Query Generator Starting...")

    # Database config
    db_config = {'host': 'localhost', 'port': '5432', 'database': 'boostsql', 'user': 'postgres', 'password': '123'}

    try:
        generator = SimpleQueryGenerator(db_config)
        queries = generator.generate_queries(1250)

        # Save to file
        os.makedirs('data', exist_ok=True)
        with open('data/queries_1250.json', 'w') as f:
            json.dump(queries, f, indent=2)

        logger.info("💾 Validated queries saved to data/queries_1250.json")

        # Print summary
        complexities = {}
        for q in queries:
            comp = q['complexity']
            complexities[comp] = complexities.get(comp, 0) + 1

        logger.info("📊 Query Distribution:")
        for comp, count in complexities.items():
            logger.info(f"  {comp.capitalize()}: {count} queries")

        logger.info("✅ Database-aware query generation completed!")

    except Exception as e:
        logger.error(f"❌ Query generation failed: {e}")

if __name__ == '__main__':
    main()