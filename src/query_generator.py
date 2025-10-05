import random
import re
from typing import List, Dict
from tqdm import tqdm

class MassiveQueryGenerator:
    def __init__(self):
        self.tables = [
            'users', 'orders', 'products', 'customers', 'payments', 
            'inventory', 'categories', 'suppliers', 'employees', 
            'departments', 'sales', 'transactions', 'logs', 'profiles',
            'user_sessions', 'order_items', 'product_reviews', 'shipping'
        ]
        
        self.columns = {
            'users': ['id', 'name', 'email', 'age', 'city', 'country', 'created_at', 'status', 'reputation'],
            'orders': ['id', 'user_id', 'product_id', 'quantity', 'price', 'order_date', 'status', 'total_amount'],
            'products': ['id', 'name', 'category', 'price', 'stock', 'supplier_id', 'rating', 'description'],
            'customers': ['id', 'company_name', 'contact_name', 'city', 'country', 'phone', 'email'],
            'payments': ['id', 'order_id', 'amount', 'payment_date', 'method', 'status'],
            'user_sessions': ['id', 'user_id', 'login_time', 'logout_time', 'ip_address'],
            'order_items': ['id', 'order_id', 'product_id', 'quantity', 'unit_price'],
            'product_reviews': ['id', 'product_id', 'user_id', 'rating', 'comment', 'created_at']
        }
        
        self.functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'STRING_AGG']
        self.operators = ['=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN', 'BETWEEN']
        self.join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']
    
    def generate_500k_dataset(self) -> List[Dict]:
        """Gera 500.000 queries cobrindo TODOS os níveis de complexidade"""
        print("🚀 Gerando 500.000 queries com complexidade variada...")
        
        dataset = []
        
        # Distribuição por complexidade
        complexity_distribution = {
            'simple': 150000,    # 30% - queries básicas
            'medium': 200000,    # 40% - queries intermediárias  
            'complex': 100000,   # 20% - queries complexas
            'expert': 50000      # 10% - queries expert
        }
        
        for complexity, count in complexity_distribution.items():
            print(f"📊 Gerando {count} queries {complexity}...")
            for i in tqdm(range(count), desc=f"{complexity}"):
                query = self._generate_query_by_complexity(complexity)
                exec_time = self._calculate_execution_time(query, complexity)
                
                dataset.append({
                    'query_id': len(dataset) + 1,
                    'query_sql': query,
                    'execution_time_ms': exec_time,
                    'complexity': complexity
                })
        
        print(f"✅ Dataset gerado: {len(dataset):,} queries")
        return dataset
    
    def _generate_query_by_complexity(self, complexity: str) -> str:
        """Gera query baseada no nível de complexidade"""
        if complexity == 'simple':
            return self._generate_simple_query()
        elif complexity == 'medium':
            return self._generate_medium_query()
        elif complexity == 'complex':
            return self._generate_complex_query()
        else:  # expert
            return self._generate_expert_query()
    
    def _generate_simple_query(self) -> str:
        """Queries simples: SELECT básico, 1 tabela"""
        templates = [
            "SELECT {columns} FROM {table} {where}",
            "SELECT {columns} FROM {table} {where} ORDER BY {order_column}",
            "SELECT {columns} FROM {table} {where} LIMIT {limit}",
            "SELECT COUNT(*) FROM {table} {where}",
            "SELECT DISTINCT {column} FROM {table}"
        ]
        
        table = random.choice(self.tables)
        template = random.choice(templates)
        
        return template.format(
            columns=self._select_columns(table, 1, 3),
            table=table,
            where=self._generate_where_clause(table, 1),
            order_column=random.choice(self.columns.get(table, ['id'])),
            limit=random.randint(10, 100),
            column=random.choice(self.columns.get(table, ['id']))
        )
    
    def _generate_medium_query(self) -> str:
        """Queries médias: JOINs simples, agregações básicas"""
        templates = [
            "SELECT {columns} FROM {table1} JOIN {table2} ON {join_condition} {where}",
            "SELECT {function}({column}) FROM {table} {where}",
            "SELECT {columns}, COUNT(*) FROM {table} {where} GROUP BY {columns}",
            "SELECT {columns} FROM {table} {where} ORDER BY {order_column} LIMIT {limit}",
            "SELECT {columns} FROM {table} WHERE {column} IN (SELECT {sub_column} FROM {sub_table} WHERE {sub_where})"
        ]
        
        table1, table2 = random.sample(self.tables, 2)
        template = random.choice(templates)
        
        return template.format(
            columns=self._select_columns(table1, 1, 4),
            table1=table1,
            table2=table2,
            table=table1,
            join_condition=f"{table1}.id = {table2}.{table1[:-1]}_id",
            where=self._generate_where_clause(table1, 2),
            function=random.choice(self.functions),
            column=random.choice(self.columns.get(table1, ['id'])),
            order_column=random.choice(self.columns.get(table1, ['id'])),
            limit=random.randint(5, 50),
            sub_column=random.choice(self.columns.get(table2, ['id'])),
            sub_table=table2,
            sub_where=self._generate_simple_condition(table2)
        )
    
    def _generate_complex_query(self) -> str:
        """Queries complexas: Múltiplos JOINs, subqueries, agregações"""
        templates = [
            """SELECT {t1_cols}, {t2_cols}, {function}({metric}) as calculated_value
            FROM {table1} 
            JOIN {table2} ON {join_condition1}
            JOIN {table3} ON {join_condition2}
            {where} 
            GROUP BY {t1_cols}, {t2_cols}
            HAVING {function}({metric}) > {threshold}""",
            
            """WITH cte AS (
                SELECT {columns}, COUNT(*) as count_value 
                FROM {table} 
                {where} 
                GROUP BY {columns}
            )
            SELECT * FROM cte WHERE count_value > {min_count} ORDER BY count_value DESC""",
            
            """SELECT {columns} 
            FROM {table1} 
            WHERE {table1}.id IN (
                SELECT {table2}.{table1_id}_id 
                FROM {table2} 
                WHERE {sub_where}
                GROUP BY {table2}.{table1_id}_id 
                HAVING COUNT(*) > {sub_count}
            ) {where}"""
        ]
        
        tables = random.sample(self.tables, 3)
        template = random.choice(templates)
        
        # Prepara os parâmetros
        params = {
            't1_cols': self._select_columns(tables[0], 1, 2),
            't2_cols': self._select_columns(tables[1], 1, 2),
            'table1': tables[0],
            'table2': tables[1],
            'table3': tables[2],
            'table': tables[0],
            'join_condition1': f"{tables[0]}.id = {tables[1]}.{self._get_foreign_key(tables[0])}",
            'join_condition2': f"{tables[1]}.id = {tables[2]}.{self._get_foreign_key(tables[1])}",
            'where': self._generate_where_clause(tables[0], 3),
            'function': random.choice(self.functions),
            'metric': random.choice(['price', 'quantity', 'amount', 'rating']),
            'threshold': random.randint(100, 1000),
            'columns': self._select_columns(tables[0], 1, 3),
            'min_count': random.randint(1, 10),
            'table1_id': tables[0][:-1] if tables[0].endswith('s') else tables[0],
            'sub_where': self._generate_where_clause(tables[1], 2),
            'sub_count': random.randint(1, 5)
        }
        
        return template.format(**params)

    def _get_foreign_key(self, table_name):
        """Gera nome de foreign key realista"""
        if table_name.endswith('s'):
            return table_name[:-1] + '_id'
        return table_name + '_id'
        
    def _generate_expert_query(self) -> str:
        """Queries expert: CTEs complexas, múltiplas subqueries, window functions"""
        templates = [
            """WITH 
               user_stats AS (
                   SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as total_spent
                   FROM orders 
                   WHERE order_date > '{date}'
                   GROUP BY user_id
               ),
               product_stats AS (
                   SELECT product_id, AVG(rating) as avg_rating, COUNT(*) as review_count
                   FROM product_reviews
                   GROUP BY product_id
                   HAVING COUNT(*) > {min_reviews}
               )
               SELECT 
                   u.name,
                   us.order_count,
                   us.total_spent,
                   p.name as product_name,
                   ps.avg_rating
               FROM users u
               JOIN user_stats us ON u.id = us.user_id
               JOIN order_items oi ON u.id = oi.order_id
               JOIN products p ON oi.product_id = p.id
               JOIN product_stats ps ON p.id = ps.product_id
               WHERE us.total_spent > {min_spent}
               ORDER BY us.total_spent DESC
               LIMIT {limit}""",
            
            """SELECT 
                   department,
                   employee_name,
                   salary,
                   AVG(salary) OVER (PARTITION BY department) as avg_department_salary,
                   RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
               FROM employees
               WHERE hire_date > '{hire_date}'
               QUALIFY salary_rank <= {top_n}""",
            
            """SELECT 
                   DATE_TRUNC('month', order_date) as month,
                   product_category,
                   SUM(quantity) as total_sold,
                   SUM(SUM(quantity)) OVER (PARTITION BY product_category ORDER BY month) as running_total,
                   LAG(SUM(quantity), 1) OVER (PARTITION BY product_category ORDER BY month) as prev_month_sales
               FROM orders o
               JOIN order_items oi ON o.id = oi.order_id
               JOIN products p ON oi.product_id = p.id
               WHERE order_date BETWEEN '{start_date}' AND '{end_date}'
               GROUP BY month, product_category
               HAVING SUM(quantity) > {min_sold}"""
        ]
        
        template = random.choice(templates)
        
        return template.format(
            date=f"2023-{random.randint(1,12):02d}-01",
            min_reviews=random.randint(5, 20),
            min_spent=random.randint(500, 2000),
            limit=random.randint(10, 50),
            hire_date=f"202{random.randint(0,3)}-01-01",
            top_n=random.randint(3, 10),
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_sold=random.randint(10, 100)
        )
    
    def _select_columns(self, table: str, min_cols: int, max_cols: int) -> str:
        """Seleciona colunas para a query"""
        if table in self.columns:
            available_cols = self.columns[table]
            num_cols = random.randint(min_cols, min(max_cols, len(available_cols)))
            selected_cols = random.sample(available_cols, num_cols)
            return ', '.join(selected_cols)
        return '*'
    
    def _generate_where_clause(self, table: str, max_conditions: int) -> str:
        """Gera cláusula WHERE realista"""
        if table not in self.columns or random.random() < 0.1:  # 10% sem WHERE
            return ""
        
        conditions = []
        num_conditions = random.randint(1, max_conditions)
        
        for _ in range(num_conditions):
            conditions.append(self._generate_condition(table))
        
        return "WHERE " + " AND ".join(conditions)
    
    def _generate_simple_condition(self, table: str) -> str:
        """Gera condição simples para subqueries"""
        if table not in self.columns:
            return "1=1"
        
        col = random.choice(self.columns[table])
        operator = random.choice(['=', '>', '<'])
        
        if col in ['status', 'category']:
            value = f"'{random.choice(['active', 'inactive', 'pending', 'completed'])}'"
        else:
            value = str(random.randint(1, 1000))
        
        return f"{col} {operator} {value}"
    
    def _generate_condition(self, table: str) -> str:
        """Gera condição WHERE realista"""
        col = random.choice(self.columns[table])
        operator = random.choice(self.operators)
        
        if operator == 'LIKE':
            value = f"'%{random.choice(['test', 'user', 'prod', 'order', 'admin'])}%'"
        elif operator == 'IN':
            values = [str(random.randint(1, 100)) for _ in range(3)]
            value = f"({', '.join(values)})"
        elif operator == 'BETWEEN':
            value = f"{random.randint(1, 500)} AND {random.randint(501, 1000)}"
        elif col in ['status', 'category', 'type']:
            value = f"'{random.choice(['active', 'inactive', 'pending', 'completed', 'cancelled'])}'"
        elif col in ['name', 'email', 'description']:
            value = f"'%{random.choice(['john', 'test', 'example', 'demo'])}%'"
            operator = 'LIKE'
        else:
            value = str(random.randint(1, 1000))
        
        return f"{col} {operator} {value}"
    
    def _calculate_execution_time(self, query: str, complexity: str) -> float:
        """Calcula tempo de execução realista baseado na complexidade"""
        base_times = {
            'simple': 50,
            'medium': 150, 
            'complex': 500,
            'expert': 1200
        }
        
        base_time = base_times[complexity]
        
        # Fatores de ajuste baseados na query
        complexity_factors = {
            'JOIN': 1.5, 'GROUP BY': 1.3, 'ORDER BY': 1.2, 'WHERE': 1.1,
            'HAVING': 1.4, 'WITH': 1.8, 'SUBQUERY': 1.6, 'UNION': 1.7,
            'WINDOW': 2.0, 'OVER': 1.9, 'QUALIFY': 1.8
        }
        
        query_upper = query.upper()
        multiplier = 1.0
        
        for factor, weight in complexity_factors.items():
            count = query_upper.count(factor)
            multiplier *= (weight ** min(count, 5))
        
        # Fator de tamanho
        size_factor = min(len(query) / 1000, 3.0)
        multiplier *= (1 + size_factor * 0.3)
        
        # Variação aleatória
        variation = random.uniform(0.7, 1.3)
        
        return base_time * multiplier * variation