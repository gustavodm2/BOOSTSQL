import random
import re

# Define tables and their columns based on create_tables.sql
TABLES = {
    'categories': ['id', 'name', 'description'],
    'suppliers': ['id', 'name', 'contact_name', 'city', 'country', 'phone', 'email'],
    'products': ['id', 'name', 'category', 'price', 'stock', 'supplier_id', 'rating', 'description', 'category_id'],
    'customers': ['id', 'company_name', 'contact_name', 'city', 'country', 'phone', 'email'],
    'users': ['id', 'name', 'email', 'age', 'city', 'country', 'created_at', 'status', 'reputation'],
    'orders': ['id', 'user_id', 'customer_id', 'order_date', 'status', 'total_amount'],
    'order_items': ['id', 'order_id', 'product_id', 'quantity', 'unit_price'],
    'payments': ['id', 'order_id', 'amount', 'payment_date', 'method', 'status'],
    'user_sessions': ['id', 'user_id', 'login_time', 'logout_time', 'ip_address'],
    'product_reviews': ['id', 'product_id', 'user_id', 'rating', 'comment', 'created_at'],
    'departments': ['id', 'name', 'budget'],
    'employees': ['id', 'name', 'department_id', 'salary', 'hire_date', 'email'],
    'inventory': ['id', 'product_id', 'warehouse_location', 'quantity', 'last_updated'],
    'sales': ['id', 'product_id', 'employee_id', 'quantity', 'sale_date', 'total_amount'],
    'transactions': ['id', 'user_id', 'amount', 'transaction_date', 'type', 'status'],
    'logs': ['id', 'user_id', 'action', 'timestamp', 'ip_address', 'details'],
    'profiles': ['id', 'user_id', 'bio', 'avatar_url', 'preferences'],
    'shipping': ['id', 'order_id', 'tracking_number', 'carrier', 'status', 'shipped_date', 'delivered_date']
}

def get_random_table():
    return random.choice(list(TABLES.keys()))

def get_random_column(table):
    return random.choice(TABLES[table])

def generate_select_star(table):
    return f"SELECT * FROM {table}"

def generate_tautological_where(table):
    col = 'id'  # Use id which exists in all tables
    return f"SELECT * FROM {table} WHERE {col} = {col}"

def generate_duplicate_conditions(table):
    col = 'id'
    return f"SELECT * FROM {table} WHERE {col} > 0 AND {col} > 0"

def generate_or_condition(table):
    col1 = 'id'
    col2 = 'id'
    return f"SELECT * FROM {table} WHERE {col1} > 0 OR {col2} < 100"

def generate_unnecessary_subquery(table):
    sub_table = get_random_table()
    col = 'id'
    sub_col = 'id'
    return f"SELECT * FROM {table} WHERE {col} IN (SELECT {sub_col} FROM {sub_table})"

def generate_redundant_join(table1, table2):
    # Use id for joins, assuming they might relate
    return f"SELECT * FROM {table1} t1 JOIN {table2} t2 ON t1.id = t2.id JOIN {table2} t3 ON t1.id = t3.id"

def generate_unnecessary_distinct(table):
    return f"SELECT DISTINCT * FROM {table}"

def generate_function_in_where(table):
    col = 'name' if 'name' in TABLES[table] else 'id'
    if col == 'name':
        return f"SELECT * FROM {table} WHERE LENGTH({col}) > 0"
    else:
        return f"SELECT * FROM {table} WHERE {col} > 0"

def generate_redundant_group_by(table):
    col = 'id'
    return f"SELECT {col}, COUNT(*) FROM {table} GROUP BY {col}, {col}"

def generate_redundant_order_by(table):
    col = 'id'
    return f"SELECT * FROM {table} ORDER BY {col}, {col}"

def generate_deeply_nested_subquery(table):
    sub1 = get_random_table()
    sub2 = get_random_table()
    col = 'id'
    return f"SELECT * FROM {table} WHERE {col} IN (SELECT id FROM {sub1} WHERE id IN (SELECT id FROM {sub2}))"

def generate_queries():
    queries = []
    generators = [
        generate_select_star,
        generate_tautological_where,
        generate_duplicate_conditions,
        generate_or_condition,
        generate_unnecessary_subquery,
        generate_redundant_join,
        generate_unnecessary_distinct,
        generate_function_in_where,
        generate_redundant_group_by,
        generate_redundant_order_by,
        generate_deeply_nested_subquery
    ]

    for _ in range(500):
        table = get_random_table()
        generator = random.choice(generators)
        if generator == generate_redundant_join:
            table2 = get_random_table()
            query = generator(table, table2)
        else:
            query = generator(table)
        queries.append(query)

    return queries

if __name__ == "__main__":
    queries = generate_queries()
    with open('unoptimized_queries.sql', 'w') as f:
        for query in queries:
            f.write(query + '\n')