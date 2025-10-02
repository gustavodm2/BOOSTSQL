import random
import string
from typing import List, Dict

class MassiveQueryGenerator:
    def __init__(self):
        self.tables = [
            'users', 'orders', 'products', 'customers', 'payments', 
            'inventory', 'categories', 'suppliers', 'employees', 
            'departments', 'sales', 'transactions', 'logs', 'profiles'
        ]
        
        # CORREÇÃO: Adicionei todas as tabelas no dicionário de colunas
        self.columns = {
            'users': ['id', 'name', 'email', 'age', 'city', 'country', 'created_at', 'status'],
            'orders': ['id', 'user_id', 'product_id', 'quantity', 'price', 'order_date', 'status'],
            'products': ['id', 'name', 'category', 'price', 'stock', 'supplier_id', 'rating'],
            'customers': ['id', 'company_name', 'contact_name', 'city', 'country', 'phone'],
            'payments': ['id', 'order_id', 'amount', 'payment_date', 'method', 'status'],
            'inventory': ['id', 'product_id', 'quantity', 'location', 'last_updated'],
            'categories': ['id', 'name', 'description', 'parent_id'],
            'suppliers': ['id', 'name', 'contact_email', 'city', 'country'],
            'employees': ['id', 'name', 'department_id', 'salary', 'hire_date'],
            'departments': ['id', 'name', 'manager_id', 'budget'],
            'sales': ['id', 'product_id', 'quantity', 'sale_date', 'customer_id', 'amount'],
            'transactions': ['id', 'user_id', 'amount', 'transaction_date', 'type', 'status'],
            'logs': ['id', 'user_id', 'action', 'timestamp', 'ip_address'],
            'profiles': ['id', 'user_id', 'bio', 'avatar_url', 'website']
        }
        
        self.functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
        self.operators = ['=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN']
        self.join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN']
    
    def generate_dataset(self, num_queries: int = 1000) -> List[Dict]:
        """Gera um dataset massivo de queries para treinamento"""
        dataset = []
        
        for i in range(num_queries):
            complexity = random.choice(['simple', 'medium', 'complex'])
            query = self._generate_realistic_query(complexity)
            
            # Simula tempo de execução baseado na complexidade
            base_time = {
                'simple': random.randint(10, 100),
                'medium': random.randint(100, 500),
                'complex': random.randint(500, 2000)
            }[complexity]
            
            # Adiciona variação aleatória
            execution_time = base_time * random.uniform(0.8, 1.2)
            
            dataset.append({
                'query_id': i + 1,
                'query_sql': query,
                'execution_time_ms': execution_time,
                'complexity': complexity
            })
            
            if (i + 1) % 100 == 0:
                print(f"Geradas {i + 1} queries...")
        
        return dataset
    
    def _generate_realistic_query(self, complexity: str) -> str:
        """Gera uma query SQL realista baseada na complexidade"""
        
        if complexity == 'simple':
            return self._generate_simple_query()
        elif complexity == 'medium':
            return self._generate_medium_query()
        else:
            return self._generate_complex_query()
    
    def _generate_simple_query(self) -> str:
        """Gera queries simples (1 tabela, poucas condições)"""
        # CORREÇÃO: Só usa tabelas que existem no dicionário
        available_tables = list(self.columns.keys())
        table = random.choice(available_tables)
        columns = self._select_columns(table, 1, 3)
        where_clause = self._generate_where_clause(table, 1)
        
        templates = [
            f"SELECT {columns} FROM {table} {where_clause}",
            f"SELECT {columns} FROM {table} {where_clause} ORDER BY {random.choice(self.columns[table])}",
            f"SELECT {columns} FROM {table} {where_clause} LIMIT {random.randint(1, 100)}"
        ]
        
        return random.choice(templates)
    
    def _generate_medium_query(self) -> str:
        """Gera queries médias (joins simples, agregações)"""
        available_tables = list(self.columns.keys())
        table1, table2 = random.sample(available_tables, 2)
        columns = self._select_columns(table1, 1, 2)
        
        templates = [
            f"SELECT {columns}, COUNT(*) FROM {table1} GROUP BY {columns}",
            f"SELECT {columns} FROM {table1} JOIN {table2} ON {table1}.id = {table2}.{table1[:-1]}_id {self._generate_where_clause(table1, 2)}",
            f"SELECT AVG({random.choice(self.columns[table1])}) FROM {table1} {self._generate_where_clause(table1, 2)}"
        ]
        
        return random.choice(templates)
    
    def _generate_complex_query(self) -> str:
        """Gera queries complexas (múltiplos joins, subqueries)"""
        available_tables = list(self.columns.keys())
        tables = random.sample(available_tables, min(3, len(available_tables)))
        function = random.choice(self.functions)
        
        templates = [
            # Query com múltiplos JOINs
            f"""SELECT {tables[0]}.name, {tables[1]}.amount, {tables[2]}.category 
                FROM {tables[0]} 
                JOIN {tables[1]} ON {tables[0]}.id = {tables[1]}.{tables[0][:-1]}_id 
                JOIN {tables[2]} ON {tables[1]}.product_id = {tables[2]}.id 
                {self._generate_where_clause(tables[0], 2)} 
                ORDER BY {tables[1]}.amount DESC 
                LIMIT 50""",
            
            # Query com subquery
            f"""SELECT name, email FROM {tables[0]} 
                WHERE id IN (
                    SELECT {tables[0][:-1]}_id FROM {tables[1]} 
                    WHERE amount > {random.randint(100, 1000)}
                ) 
                {self._generate_where_clause(tables[0], 1)}""",
            
            # Query com agregação complexa
            f"""SELECT category, {function}(price) as avg_price, COUNT(*) as total 
                FROM {tables[0]} 
                {self._generate_where_clause(tables[0], 2)} 
                GROUP BY category 
                HAVING COUNT(*) > {random.randint(1, 10)} 
                ORDER BY avg_price DESC"""
        ]
        
        return random.choice(templates)
    
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
        if table not in self.columns or random.random() < 0.3:  # 30% sem WHERE
            return ""
        
        conditions = []
        num_conditions = random.randint(1, max_conditions)
        
        for _ in range(num_conditions):
            col = random.choice(self.columns[table])
            operator = random.choice(self.operators)
            
            if operator == 'LIKE':
                value = f"'%{random.choice(['test', 'user', 'prod', 'order'])}%'"
            elif operator == 'IN':
                values = [str(random.randint(1, 100)) for _ in range(3)]
                value = f"({', '.join(values)})"
            else:
                value = f"'{random.choice(['active', 'pending', 'completed'])}'" if col == 'status' else str(random.randint(1, 1000))
            
            conditions.append(f"{col} {operator} {value}")
        
        return "WHERE " + " AND ".join(conditions)