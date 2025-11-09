#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_agent import SQLBoostMLAgent

def test_agent_optimization():
    # Mock DB config
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'test',
        'user': 'test',
        'password': 'test'
    }

    agent = SQLBoostMLAgent(db_config)

    # Test query
    query = "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)"

    print("Testing agent optimization...")
    print("Original query:")
    print(query)
    print()

    # Test direct rewriter call
    rewritten = agent.query_rewriter.rewrite_query(query, 'subquery_to_join')
    print("Direct rewriter result:")
    for q in rewritten:
        print(f"  {q}")
    print()

    result = agent.optimize_query(query)

    print("Optimization result:")
    print(f"Original query: {result['original_query']}")
    print(f"Performance comparison: {result['performance_comparison']}")
    print(f"Candidates evaluated: {result['all_candidates_evaluated']}")
    print(f"Successful optimizations: {result['successful_optimizations']}")
    print()

    if result['best_optimization']:
        best = result['best_optimization']
        print("Best optimization:")
        print(f"Optimized query: {best['optimized_query']}")
        print(f"Improvement ratio: {best['improvement_ratio']:.2f}x")
        print(f"Type: {best['optimization_type']}")
    else:
        print("No optimization found")

    print()
    print("All candidates:")
    for i, candidate in enumerate(result['candidates'], 1):
        print(f"{i}. {candidate['query']} (type: {candidate['type']})")

if __name__ == "__main__":
    test_agent_optimization()