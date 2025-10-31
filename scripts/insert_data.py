import psycopg2
import random
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import os

def insert_sample_data():
    print("Inserting 100,000+ rows of sample data into SQLBoost database...")

    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'boostsql',
        'user': 'postgres',
        'password': '123'
    }

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        print("Inserting sample data...")

        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'Chris', 'Lisa', 'Mark', 'Anna', 'James', 'Mary', 'Robert', 'Patricia', 'Jennifer', 'Linda', 'William', 'Elizabeth', 'Richard', 'Barbara']
        last_names = ['Smith', 'Johnson', 'Brown', 'Williams', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin']
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Boston']
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Italy', 'Spain', 'Australia', 'Japan', 'Brazil']
        categories_list = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Toys', 'Automotive', 'Health', 'Garden']
        product_names = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Smartphone', 'Tablet', 'Printer', 'Router', 'Camera', 'Smart Watch', 'External Drive', 'USB Cable', 'Power Bank', 'Bluetooth Speaker']
        departments_list = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'Support', 'Research']
        review_comments = ['Great product!', 'Excellent quality', 'Fast shipping', 'Good value', 'Highly recommended', 'Works perfectly', 'Five stars!', 'Amazing product', 'Very satisfied', 'Will buy again', 'Good customer service', 'As described', 'Better than expected', 'Quality is excellent', 'Fast delivery']

        print("   - Inserting categories...")
        for cat in categories_list:
            cursor.execute("INSERT INTO categories (name, description) VALUES (%s, %s)",
                         (cat, f"{cat} products and accessories"))

        print("   - Inserting suppliers (5,000)...")
        for i in tqdm(range(5000), desc="Suppliers"):
            name = f"Supplier {i+1}"
            contact = f"{random.choice(first_names)} {random.choice(last_names)}"
            cursor.execute("""
                INSERT INTO suppliers (name, contact_name, city, country, phone, email)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (name, contact, random.choice(cities), random.choice(countries),
                  f"+1-{random.randint(100,999)}-{random.randint(1000,9999)}",
                  f"contact@supplier{i+1}.com"))

        print("   - Inserting departments...")
        for dept in departments_list:
            cursor.execute("INSERT INTO departments (name, budget) VALUES (%s, %s)",
                         (dept, random.randint(50000, 500000)))

        print("   - Inserting users (25,000,000)...")
        for i in tqdm(range(25000000), desc="Users"):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            email = f"user{i+1}@example.com"
            age = random.randint(18, 80)
            city = random.choice(cities)
            country = random.choice(countries)
            created_at = datetime.now() - timedelta(days=random.randint(0, 365*2))
            status = random.choice(['active', 'inactive', 'pending'])
            reputation = random.randint(0, 1000)

            cursor.execute("""
                INSERT INTO users (name, email, age, city, country, created_at, status, reputation)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (name, email, age, city, country, created_at, status, reputation))

        print("   - Inserting products (5,000)...")
        for i in tqdm(range(1000000), desc="Products"):
            name = f"{random.choice(product_names)} {random.choice(['Pro', 'Max', 'Ultra', 'Mini', 'Plus', 'X', 'Z', 'Air', 'Go', 'Lite'])}"
            category = random.choice(categories_list)
            price = round(random.uniform(10, 2000), 2)
            stock = random.randint(0, 1000)
            supplier_id = random.randint(1, 1000000)
            rating = round(random.uniform(1, 5), 1)
            description = f"High quality {category.lower()} product"
            category_id = random.randint(1, len(categories_list))

            cursor.execute("""
                INSERT INTO products (name, category, price, stock, supplier_id, rating, description, category_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (name, category, price, stock, supplier_id, rating, description, category_id))

        print("   - Inserting customers (2,000)...")
        for i in tqdm(range(5000000), desc="Customers"):
            company_name = f"Company {i+1}" if random.random() > 0.3 else None
            contact_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            city = random.choice(cities)
            country = random.choice(countries)
            phone = f"+1-{random.randint(100,999)}-{random.randint(1000,9999)}"
            email = f"customer{i+1}@example.com"

            cursor.execute("""
                INSERT INTO customers (company_name, contact_name, city, country, phone, email)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (company_name, contact_name, city, country, phone, email))

        print("   - Inserting employees (500)...")
        for i in tqdm(range(100000), desc="Employees"):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            dept_id = random.randint(1, len(departments_list))
            salary = round(random.uniform(30000, 150000), 2)
            hire_date = datetime.now() - timedelta(days=random.randint(30, 365*5))
            email = f"employee{i+1}@company.com"

            cursor.execute("""
                INSERT INTO employees (name, department_id, salary, hire_date, email)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, dept_id, salary, hire_date.date(), email))

        print("   - Inserting orders and order items (25,000 orders)...")
        for i in tqdm(range(6000000), desc="Orders"):
            user_id = random.randint(1, 25000000)
            customer_id = random.randint(1, 5000000)
            order_date = datetime.now() - timedelta(days=random.randint(0, 365))
            status = random.choice(['pending', 'processing', 'shipped', 'delivered', 'cancelled'])

            cursor.execute("""
                INSERT INTO orders (user_id, customer_id, order_date, status)
                VALUES (%s, %s, %s, %s) RETURNING id
            """, (user_id, customer_id, order_date, status))

            order_id = cursor.fetchone()[0]

            num_items = random.randint(1, 5)
            total_amount = 0

            for _ in range(num_items):
                product_id = random.randint(1, 1000000)
                quantity = random.randint(1, 10)

                cursor.execute("SELECT price FROM products WHERE id = %s", (product_id,))
                unit_price = cursor.fetchone()[0]

                cursor.execute("""
                    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                    VALUES (%s, %s, %s, %s)
                """, (order_id, product_id, quantity, unit_price))

                total_amount += unit_price * quantity

            cursor.execute("UPDATE orders SET total_amount = %s WHERE id = %s",
                         (total_amount, order_id))

        print("   - Inserting payments (25,000)...")
        cursor.execute("SELECT id FROM orders")
        order_ids = [row[0] for row in cursor.fetchall()]

        for order_id in tqdm(order_ids, desc="Payments"):
            cursor.execute("SELECT total_amount FROM orders WHERE id = %s", (order_id,))
            amount = cursor.fetchone()[0]
            payment_date = datetime.now() - timedelta(days=random.randint(0, 30))
            method = random.choice(['credit_card', 'paypal', 'bank_transfer', 'cash'])
            status = random.choice(['completed', 'pending', 'failed'])

            cursor.execute("""
                INSERT INTO payments (order_id, amount, payment_date, method, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (order_id, amount, payment_date, method, status))

        print("   - Inserting user sessions (50,000)...")
        for i in tqdm(range(12000000), desc="Sessions"):
            user_id = random.randint(1, 25000000)
            login_time = datetime.now() - timedelta(days=random.randint(0, 30))
            logout_time = login_time + timedelta(minutes=random.randint(5, 480)) if random.random() > 0.1 else None
            ip_address = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"

            cursor.execute("""
                INSERT INTO user_sessions (user_id, login_time, logout_time, ip_address)
                VALUES (%s, %s, %s, %s)
            """, (user_id, login_time, logout_time, ip_address))

        print("   - Inserting product reviews (15,000)...")
        for i in tqdm(range(4000000), desc="Reviews"):
            product_id = random.randint(1, 1000000)
            user_id = random.randint(1, 25000000)
            rating = random.randint(1, 5)
            comment = random.choice(review_comments)
            created_at = datetime.now() - timedelta(days=random.randint(0, 365))

            cursor.execute("""
                INSERT INTO product_reviews (product_id, user_id, rating, comment, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (product_id, user_id, rating, comment, created_at))

        print("   - Inserting inventory records (5,000)...")
        for i in tqdm(range(1000000), desc="Inventory"):
            product_id = random.randint(1, 1000000)
            warehouse_location = f"Warehouse-{random.choice(['A', 'B', 'C'])}-{random.randint(1,10)}"
            quantity = random.randint(0, 500)
            last_updated = datetime.now() - timedelta(days=random.randint(0, 30))

            cursor.execute("""
                INSERT INTO inventory (product_id, warehouse_location, quantity, last_updated)
                VALUES (%s, %s, %s, %s)
            """, (product_id, warehouse_location, quantity, last_updated))

        print("   - Inserting sales records (10,000)...")
        for i in tqdm(range(2000000), desc="Sales"):
            product_id = random.randint(1, 1000000)
            employee_id = random.randint(1, 100000)
            quantity = random.randint(1, 20)
            sale_date = datetime.now() - timedelta(days=random.randint(0, 365))

            cursor.execute("SELECT price FROM products WHERE id = %s", (product_id,))
            unit_price = cursor.fetchone()[0]
            total_amount = unit_price * quantity

            cursor.execute("""
                INSERT INTO sales (product_id, employee_id, quantity, sale_date, total_amount)
                VALUES (%s, %s, %s, %s, %s)
            """, (product_id, employee_id, quantity, sale_date, total_amount))

        print("   - Inserting transactions (30,000)...")
        for i in tqdm(range(7000000), desc="Transactions"):
            user_id = random.randint(1, 25000000)
            amount = round(random.uniform(10, 5000), 2)
            transaction_date = datetime.now() - timedelta(days=random.randint(0, 365))
            type_ = random.choice(['purchase', 'refund', 'deposit', 'withdrawal'])
            status = random.choice(['completed', 'pending', 'failed'])

            cursor.execute("""
                INSERT INTO transactions (user_id, amount, transaction_date, type, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, amount, transaction_date, type_, status))

        print("   - Inserting logs (20,000)...")
        for i in tqdm(range(5000000), desc="Logs"):
            user_id = random.randint(1, 25000000)
            action = random.choice(['login', 'logout', 'purchase', 'view_product', 'add_to_cart', 'checkout'])
            timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
            ip_address = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
            details = f"User performed {action}"

            cursor.execute("""
                INSERT INTO logs (user_id, action, timestamp, ip_address, details)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, action, timestamp, ip_address, details))

        print("   - Inserting profiles (10,000)...")
        for i in tqdm(range(25000000), desc="Profiles"):
            user_id = i + 1
            bio = f"Bio for user {user_id}"
            avatar_url = f"https://example.com/avatar/{user_id}.jpg"
            preferences = '{"theme": "dark", "notifications": true}'

            cursor.execute("""
                INSERT INTO profiles (user_id, bio, avatar_url, preferences)
                VALUES (%s, %s, %s, %s)
            """, (user_id, bio, avatar_url, preferences))

        print("   - Inserting shipping records (25,000)...")
        cursor.execute("SELECT id FROM orders")
        order_ids = [row[0] for row in cursor.fetchall()]

        for order_id in tqdm(order_ids, desc="Shipping"):
            tracking_number = f"TN{random.randint(100000000, 999999999)}"
            carrier = random.choice(['UPS', 'FedEx', 'USPS', 'DHL'])
            status = random.choice(['pending', 'shipped', 'in_transit', 'delivered'])
            shipped_date = datetime.now() - timedelta(days=random.randint(0, 14)) if status != 'pending' else None
            delivered_date = shipped_date + timedelta(days=random.randint(1, 7)) if shipped_date and status == 'delivered' else None

            cursor.execute("""
                INSERT INTO shipping (order_id, tracking_number, carrier, status, shipped_date, delivered_date)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (order_id, tracking_number, carrier, status, shipped_date, delivered_date))

        conn.commit()
        conn.close()

        print("\n🎉 Data insertion complete!")
        print("📊 Summary:")
        print("   - 10,000 users")
        print("   - 5,000 products")
        print("   - 25,000 orders")
        print("   - 50,000+ order items")
        print("   - 50,000 user sessions")
        print("   - 15,000 product reviews")
        print("   - And more...")
        print("   - Total: ~100,000+ rows across all tables")
        print("\n🚀 You can now run: python scripts/run_real_ml_agent.py")

    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    insert_sample_data()