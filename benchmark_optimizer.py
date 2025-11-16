import psycopg2
import csv
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ml_agent import SQLBoostMLAgent

DB_CONFIG = {
    'host': 'localhost',
    'database': 'boostsql',
    'user': 'postgres',
    'password': '123',
    'port': 5432
}

def parse_explain_analyze(output):
    """Parse EXPLAIN ANALYZE output to extract metrics"""
    lines = output.split('\n')
    execution_time = 0
    total_cost = 0
    rows_read = 0
    loops = 0

    for line in lines:
        time_match = re.search(r'Execution Time: (\d+\.?\d*) ms', line)
        if time_match:
            execution_time = float(time_match.group(1))

        # Extract cost (total cost)
        cost_match = re.search(r'cost=\d+\.?\d*\.\.(\d+\.?\d*)', line)
        if cost_match:
            total_cost = float(cost_match.group(1))

        # Extract rows
        rows_match = re.search(r'rows=(\d+)', line)
        if rows_match:
            rows_read = int(rows_match.group(1))

        # Extract loops
        loops_match = re.search(r'loops=(\d+)', line)
        if loops_match:
            loops = int(loops_match.group(1))

    plan_metrics = f"cost={total_cost}, rows={rows_read}, loops={loops}"
    return execution_time, plan_metrics

def execute_explain_analyze(cursor, query):
    """Execute EXPLAIN ANALYZE and return parsed metrics"""
    explain_query = f"EXPLAIN ANALYZE {query}"
    cursor.execute(explain_query)
    output = '\n'.join(row[0] for row in cursor.fetchall())
    return parse_explain_analyze(output)

def main():
    # Initialize the optimizer
    agent = SQLBoostMLAgent(DB_CONFIG)

    # Define optimize_sql function
    def optimize_sql(query):
        return agent.optimize_query_simple(query)

    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()

    # Load queries
    with open('unoptimized_queries.sql', 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    results = []

    for i, original_query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}")

        try:
            # Get optimized query
            optimized_query = optimize_sql(original_query)

            # Execute original
            orig_time, orig_metrics = execute_explain_analyze(cursor, original_query)

            # Execute optimized
            opt_time, opt_metrics = execute_explain_analyze(cursor, optimized_query)

            results.append({
                'original_query': original_query,
                'optimized_query': optimized_query,
                'original_execution_time': orig_time,
                'optimized_execution_time': opt_time,
                'original_plan_metrics': orig_metrics,
                'optimized_plan_metrics': opt_metrics
            })

        except Exception as e:
            print(f"Error processing query {i+1}: {e}")
            continue

    # Close connection
    cursor.close()
    conn.close()

    # Save to CSV
    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['original_query', 'optimized_query', 'original_execution_time', 'optimized_execution_time', 'original_plan_metrics', 'optimized_plan_metrics']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Benchmarking complete. Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()