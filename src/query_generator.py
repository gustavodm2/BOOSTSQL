import random
import re
from typing import List, Dict
from tqdm import tqdm

class MassiveQueryGenerator:

    def __init__(self):
        # These will be populated from the actual database schema
        self.tables = []
        self.columns = {}
        self.column_types = {}
        self.foreign_keys = {}

        # Valid operators for different data types
        self.numeric_operators = ['=', '>', '<', '>=', '<=', '!=', 'IN', 'BETWEEN']
        self.string_operators = ['=', '!=', 'LIKE', 'IN', 'BETWEEN']
        self.date_operators = ['=', '>', '<', '>=', '<=', '!=', 'BETWEEN']
        self.functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
        self.join_types = ['INNER JOIN', 'LEFT JOIN']

    def adapt_to_database(self, db_connector):
        """Adapt the query generator to the actual database schema"""
        try:
            schema = db_connector.discover_schema()
            self.tables = list(schema.keys())
            self.columns = schema

            # Get column types and foreign keys
            self._analyze_schema(db_connector)

            print(f"Adapted to database with {len(self.tables)} tables: {self.tables}")
        except Exception as e:
            print(f"Error adapting to database: {e}")
            # Fallback to basic schema
            self._fallback_schema()

    def _analyze_schema(self, db_connector):
        """Analyze the database schema to understand column types and relationships"""
        try:
            with db_connector.get_connection() as conn:
                cursor = conn.cursor()

                # Get column information
                cursor.execute("""
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """)

                for row in cursor.fetchall():
                    table_name, column_name, data_type, is_nullable = row
                    if table_name not in self.column_types:
                        self.column_types[table_name] = {}
                    self.column_types[table_name][column_name] = {
                        'type': data_type,
                        'nullable': is_nullable == 'YES'
                    }

                # Get foreign key relationships
                cursor.execute("""
                    SELECT
                        tc.table_name, kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                      AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = 'public'
                """)

                for row in cursor.fetchall():
                    table_name, column_name, foreign_table, foreign_column = row
                    if table_name not in self.foreign_keys:
                        self.foreign_keys[table_name] = {}
                    self.foreign_keys[table_name][column_name] = {
                        'table': foreign_table,
                        'column': foreign_column
                    }

        except Exception as e:
            print(f"Error analyzing schema: {e}")

    def _fallback_schema(self):
        """Fallback schema if database analysis fails"""
        self.tables = ['users', 'orders', 'products', 'customers', 'payments', 'user_sessions', 'order_items', 'product_reviews', 'categories', 'suppliers']
        self.columns = {
            'users': ['id', 'name', 'email', 'age', 'city', 'country', 'created_at', 'status', 'reputation'],
            'orders': ['id', 'user_id', 'customer_id', 'order_date', 'status', 'total_amount'],
            'products': ['id', 'name', 'category', 'price', 'stock', 'supplier_id', 'rating', 'description', 'category_id'],
            'customers': ['id', 'company_name', 'contact_name', 'city', 'country', 'phone', 'email'],
            'payments': ['id', 'order_id', 'amount', 'payment_date', 'method', 'status'],
            'user_sessions': ['id', 'user_id', 'login_time', 'logout_time', 'ip_address'],
            'order_items': ['id', 'order_id', 'product_id', 'quantity', 'unit_price'],
            'product_reviews': ['id', 'product_id', 'user_id', 'rating', 'comment', 'created_at'],
            'categories': ['id', 'name', 'description'],
            'suppliers': ['id', 'name', 'contact_name', 'city', 'country', 'phone', 'email']
        }

    def generate_500k_dataset(self) -> List[Dict]:
        print('Gerando 500.000 queries com complexidade variada...')
        dataset = []
        complexity_distribution = {'simple': 150000, 'medium': 200000, 'complex': 100000, 'expert': 50000}
        for complexity, count in complexity_distribution.items():
            print(f'Gerando {count} queries {complexity}...')
            for i in tqdm(range(count), desc=f'{complexity}'):
                query = self._generate_query_by_complexity(complexity)
                exec_time = self._calculate_execution_time(query, complexity)
                dataset.append({'query_id': len(dataset) + 1, 'query_sql': query, 'execution_time_ms': exec_time, 'complexity': complexity})
        print(f'Dataset gerado: {len(dataset):,} queries')
        return dataset

    def _generate_query_by_complexity(self, complexity: str) -> str:
        if complexity == 'simple':
            return self._generate_simple_query()
        elif complexity == 'medium':
            return self._generate_medium_query()
        elif complexity == 'complex':
            return self._generate_complex_query()
        else:
            return self._generate_expert_query()

    def _generate_simple_query(self) -> str:
        templates = ['SELECT {columns} FROM {table} {where}', 'SELECT {columns} FROM {table} {where} ORDER BY {order_column}', 'SELECT {columns} FROM {table} {where} LIMIT {limit}', 'SELECT COUNT(*) FROM {table} {where}', 'SELECT DISTINCT {column} FROM {table}']
        table = random.choice(self.tables)
        template = random.choice(templates)
        return template.format(columns=self._select_columns(table, 1, 3), table=table, where=self._generate_where_clause(table, 1), order_column=random.choice(self.columns.get(table, ['id'])), limit=random.randint(10, 100), column=random.choice(self.columns.get(table, ['id'])))

    def _generate_medium_query(self) -> str:
        templates = [
            'SELECT {columns} FROM {table} {where}',
            'SELECT {function}({column}) FROM {table} {where}',
            'SELECT {columns}, COUNT(*) FROM {table} {where} GROUP BY {columns}',
            'SELECT {columns} FROM {table} {where} ORDER BY {order_column} LIMIT {limit}',
            'SELECT {columns} FROM {table} WHERE {column} IN (SELECT {sub_column} FROM {sub_table} WHERE {sub_where})'
        ]

        # For joins, try to find valid relationships
        join_templates = []
        if self.foreign_keys:
            join_condition = self._get_valid_join_condition()
            if join_condition:
                join_templates = ['SELECT {columns} FROM {table1} JOIN {table2} ON {join_condition} {where}']

        all_templates = templates + join_templates

        template = random.choice(all_templates)

        if 'JOIN' in template:
            # This is a join query
            table1, table2, join_condition = self._get_valid_join_condition()
            return template.format(
                columns=self._select_columns(table1, 1, 4),
                table1=table1,
                table2=table2,
                join_condition=join_condition,
                where=self._generate_where_clause(table1, 2)
            )
        else:
            # This is a single table query
            table = random.choice(self.tables)
            return template.format(
                columns=self._select_columns(table, 1, 4),
                table=table,
                where=self._generate_where_clause(table, 2),
                function=random.choice(['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
                column=random.choice(self.columns.get(table, ['id']) if table in self.columns else ['id']),
                order_column=random.choice(self.columns.get(table, ['id']) if table in self.columns else ['id']),
                limit=random.randint(5, 50),
                sub_column=random.choice(self.columns.get(table, ['id']) if table in self.columns else ['id']),
                sub_table=table,
                sub_where=self._generate_simple_condition(table)
            )

    def _generate_complex_query(self) -> str:
        # Simpler templates to avoid complex joins that might fail
        templates = [
            'WITH cte AS (\n                SELECT {columns}, COUNT(*) as count_value \n                FROM {table} \n                {where} \n                GROUP BY {columns}\n            )\n            SELECT * FROM cte WHERE count_value > {min_count} ORDER BY count_value DESC',
            'SELECT {columns}, COUNT(*) FROM {table} {where} GROUP BY {columns} HAVING COUNT(*) > {min_count}',
            'SELECT {columns} FROM {table} {where} ORDER BY {order_col} LIMIT {limit}'
        ]

        template = random.choice(templates)
        table = random.choice(self.tables)

        return template.format(
            columns=self._select_columns(table, 1, 3),
            table=table,
            where=self._generate_where_clause(table, 2),
            min_count=random.randint(1, 10),
            order_col=random.choice(self.columns.get(table, ['id']) if table in self.columns else ['id']),
            limit=random.randint(10, 100)
        )

    def _get_foreign_key(self, table_name):
        if table_name.endswith('s'):
            return table_name[:-1] + '_id'
        return table_name + '_id'

    def _get_valid_join_condition(self):
        """Get a valid join condition based on actual foreign keys"""
        if not self.foreign_keys:
            # Fallback: try some common relationships
            possible_joins = [
                ('orders', 'users', 'orders.user_id = users.id'),
                ('orders', 'customers', 'orders.customer_id = customers.id'),
                ('order_items', 'orders', 'order_items.order_id = orders.id'),
                ('order_items', 'products', 'order_items.product_id = products.id'),
                ('payments', 'orders', 'payments.order_id = orders.id'),
                ('user_sessions', 'users', 'user_sessions.user_id = users.id'),
                ('product_reviews', 'products', 'product_reviews.product_id = products.id'),
                ('product_reviews', 'users', 'product_reviews.user_id = users.id'),
                ('products', 'categories', 'products.category_id = categories.id'),
                ('products', 'suppliers', 'products.supplier_id = suppliers.id'),
                ('employees', 'departments', 'employees.department_id = departments.id'),
                ('sales', 'products', 'sales.product_id = products.id'),
                ('sales', 'employees', 'sales.employee_id = employees.id'),
                ('inventory', 'products', 'inventory.product_id = products.id'),
                ('transactions', 'users', 'transactions.user_id = users.id'),
                ('logs', 'users', 'logs.user_id = users.id'),
                ('profiles', 'users', 'profiles.user_id = users.id'),
                ('shipping', 'orders', 'shipping.order_id = orders.id')
            ]
            return random.choice(possible_joins)

        # Find a valid foreign key relationship
        valid_tables = [t for t in self.foreign_keys.keys() if self.foreign_keys[t]]
        if not valid_tables:
            return random.choice(self.tables), random.choice(self.tables), '1=1'

        table1 = random.choice(valid_tables)
        fk_col = random.choice(list(self.foreign_keys[table1].keys()))
        fk_info = self.foreign_keys[table1][fk_col]

        table2 = fk_info['table']
        join_condition = f'{table1}.{fk_col} = {table2}.{fk_info["column"]}'

        return table1, table2, join_condition

    def _generate_expert_query(self) -> str:
        templates = ["WITH \n               user_stats AS (\n                   SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as total_spent\n                   FROM orders \n                   WHERE order_date > '{date}'\n                   GROUP BY user_id\n               ),\n               product_stats AS (\n                   SELECT product_id, AVG(rating) as avg_rating, COUNT(*) as review_count\n                   FROM product_reviews\n                   GROUP BY product_id\n                   HAVING COUNT(*) > {min_reviews}\n               )\n               SELECT \n                   u.name,\n                   us.order_count,\n                   us.total_spent,\n                   p.name as product_name,\n                   ps.avg_rating\n               FROM users u\n               JOIN user_stats us ON u.id = us.user_id\n               JOIN order_items oi ON u.id = oi.order_id\n               JOIN products p ON oi.product_id = p.id\n               JOIN product_stats ps ON p.id = ps.product_id\n               WHERE us.total_spent > {min_spent}\n               ORDER BY us.total_spent DESC\n               LIMIT {limit}", "SELECT \n                   department,\n                   employee_name,\n                   salary,\n                   AVG(salary) OVER (PARTITION BY department) as avg_department_salary,\n                   RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank\n               FROM employees\n               WHERE hire_date > '{hire_date}'\n               QUALIFY salary_rank <= {top_n}", "SELECT \n                   DATE_TRUNC('month', order_date) as month,\n                   product_category,\n                   SUM(quantity) as total_sold,\n                   SUM(SUM(quantity)) OVER (PARTITION BY product_category ORDER BY month) as running_total,\n                   LAG(SUM(quantity), 1) OVER (PARTITION BY product_category ORDER BY month) as prev_month_sales\n               FROM orders o\n               JOIN order_items oi ON o.id = oi.order_id\n               JOIN products p ON oi.product_id = p.id\n               WHERE order_date BETWEEN '{start_date}' AND '{end_date}'\n               GROUP BY month, product_category\n               HAVING SUM(quantity) > {min_sold}"]
        template = random.choice(templates)
        return template.format(date=f'2023-{random.randint(1, 12):02d}-01', min_reviews=random.randint(5, 20), min_spent=random.randint(500, 2000), limit=random.randint(10, 50), hire_date=f'202{random.randint(0, 3)}-01-01', top_n=random.randint(3, 10), start_date='2023-01-01', end_date='2023-12-31', min_sold=random.randint(10, 100))

    def _select_columns(self, table: str, min_cols: int, max_cols: int) -> str:
        if table in self.columns:
            available_cols = self.columns[table]
            num_cols = random.randint(min_cols, min(max_cols, len(available_cols)))
            selected_cols = random.sample(available_cols, num_cols)
            return ', '.join(selected_cols)
        return '*'

    def _generate_where_clause(self, table: str, max_conditions: int) -> str:
        if table not in self.columns or random.random() < 0.1:
            return ''
        conditions = []
        num_conditions = random.randint(1, max_conditions)
        for _ in range(num_conditions):
            condition = self._generate_condition(table)
            if condition:
                conditions.append(condition)
        if not conditions:
            return ''
        return 'WHERE ' + ' AND '.join(conditions)

    def _generate_simple_condition(self, table: str) -> str:
        if table not in self.columns:
            return '1=1'

        # Prefer non-ID columns for conditions
        available_cols = [col for col in self.columns[table] if col != 'id']
        if not available_cols:
            return '1=1'

        col = random.choice(available_cols)

        # Get column type information
        col_type = 'unknown'
        if table in self.column_types and col in self.column_types[table]:
            col_type = self.column_types[table][col]['type']

        # Choose appropriate operator and value
        if col_type in ['integer', 'numeric', 'decimal', 'real', 'bigint', 'smallint']:
            operator = random.choice(['=', '>', '<'])
            value = str(random.randint(1, 1000))
        elif col_type in ['timestamp without time zone', 'date']:
            operator = random.choice(['=', '>'])
            value = f"'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}'"
        elif col in ['status', 'method'] or 'status' in col:
            operator = '='
            value = f"'{random.choice(['active', 'inactive', 'pending', 'completed', 'paid', 'failed'])}'"
        elif col in ['rating']:
            operator = '='
            value = str(random.randint(1, 5))
        else:
            operator = 'LIKE'
            value = f"'%{random.choice(['test', 'user', 'example'])}%'"

        return f'{col} {operator} {value}'

    def _generate_condition(self, table: str) -> str:
        if table not in self.columns:
            return '1=1'

        # Skip ID columns for WHERE conditions
        available_cols = [col for col in self.columns[table] if col != 'id']
        if not available_cols:
            return '1=1'

        col = random.choice(available_cols)

        # Get column type information
        col_type = 'unknown'
        if table in self.column_types and col in self.column_types[table]:
            col_type = self.column_types[table][col]['type']

        # Choose appropriate operator based on column type
        if col_type in ['integer', 'numeric', 'decimal', 'real', 'double precision', 'bigint', 'smallint']:
            operator = random.choice(self.numeric_operators)
            if operator == 'LIKE':
                operator = '='  # LIKE doesn't work with numbers
        elif col_type in ['timestamp without time zone', 'timestamp with time zone', 'date', 'time']:
            operator = random.choice(self.date_operators)
            if operator == 'LIKE':
                operator = '='
        else:  # text, varchar, char
            operator = random.choice(self.string_operators)

        # Generate appropriate value based on operator and column type
        if operator == 'LIKE':
            value = f"'%{random.choice(['test', 'user', 'prod', 'order', 'admin', 'example'])}%'"
        elif operator == 'IN':
            if col_type in ['integer', 'numeric', 'decimal', 'real', 'bigint', 'smallint']:
                values = [str(random.randint(1, 100)) for _ in range(3)]
            elif col_type in ['timestamp without time zone', 'date']:
                values = [f"'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}'" for _ in range(3)]
            elif col_type == 'inet':
                values = [f"'192.168.{random.randint(1,255)}.{random.randint(1,255)}'" for _ in range(3)]
            else:
                values = [f"'{random.choice(['active', 'inactive', 'pending', 'completed'])}'" for _ in range(3)]
            value = f'({', '.join(values)})'
        elif operator == 'BETWEEN':
            if col_type in ['integer', 'numeric', 'decimal', 'real', 'bigint', 'smallint']:
                value = f'{random.randint(1, 500)} AND {random.randint(501, 1000)}'
            elif col_type in ['timestamp without time zone', 'date']:
                value = f"'2024-01-01' AND '2024-12-31'"
            elif col_type == 'inet':
                value = f"'192.168.1.1' AND '192.168.255.255'"
            else:
                value = f"'a' AND 'z'"
        elif col_type in ['integer', 'numeric', 'decimal', 'real', 'bigint', 'smallint']:
            value = str(random.randint(1, 1000))
        elif col_type in ['timestamp without time zone', 'date']:
            value = f"'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}'"
        elif col_type == 'inet':
            value = f"'192.168.{random.randint(1,255)}.{random.randint(1,255)}'"
        elif col in ['status', 'method'] or 'status' in col:
            value = f"'{random.choice(['active', 'inactive', 'pending', 'completed', 'cancelled', 'paid', 'failed'])}'"
        elif col in ['rating']:
            value = str(random.randint(1, 5))
        elif 'email' in col:
            value = f"'user{random.randint(1,10000)}@example.com'"
        elif 'name' in col or 'description' in col or 'comment' in col:
            value = f"'{random.choice(['Test item', 'Sample data', 'Example entry', 'Demo record'])}'"
        else:
            value = f"'{random.choice(['value1', 'value2', 'value3'])}'"

        return f'{col} {operator} {value}'

    def _calculate_execution_time(self, query: str, complexity: str) -> float:
        base_times = {'simple': 50, 'medium': 150, 'complex': 500, 'expert': 1200}
        base_time = base_times[complexity]
        complexity_factors = {'JOIN': 1.5, 'GROUP BY': 1.3, 'ORDER BY': 1.2, 'WHERE': 1.1, 'HAVING': 1.4, 'WITH': 1.8, 'SUBQUERY': 1.6, 'UNION': 1.7, 'WINDOW': 2.0, 'OVER': 1.9, 'QUALIFY': 1.8}
        query_upper = query.upper()
        multiplier = 1.0
        for factor, weight in complexity_factors.items():
            count = query_upper.count(factor)
            multiplier *= weight ** min(count, 5)
        size_factor = min(len(query) / 1000, 3.0)
        multiplier *= 1 + size_factor * 0.3
        variation = random.uniform(0.7, 1.3)
        return base_time * multiplier * variation
