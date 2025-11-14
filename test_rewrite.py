#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.query_rewriter import MLQueryRewriter

def test_subquery_rewrite():
    rewriter = MLQueryRewriter()

    # Comprehensive test queries
    test_queries = [
        # Expression simplification
        ("Expression simplification", "SELECT u.name, p.title FROM users u JOIN posts p ON u.id = p.author_id WHERE u.age > 21 AND u.status = 'active' AND p.published = 1 AND u.id = u.id"),

        # Subquery to JOIN - IN subquery
        ("IN subquery to JOIN", "SELECT u.name FROM users u WHERE u.id IN (SELECT p.author_id FROM posts p WHERE p.published = 1)"),

        # EXISTS subquery
        ("EXISTS subquery", "SELECT u.name FROM users u WHERE EXISTS (SELECT 1 FROM posts p WHERE p.author_id = u.id AND p.published = 1)"),

        # Correlated subquery
        ("Correlated subquery", "SELECT u.name FROM users u WHERE u.id IN (SELECT p.author_id FROM posts p WHERE p.author_id = u.id AND p.published = 1)"),

        # CTE materialization
        ("CTE materialization", "WITH user_posts AS (SELECT u.id, COUNT(p.id) as post_count FROM users u LEFT JOIN posts p ON u.id = p.author_id GROUP BY u.id) SELECT * FROM user_posts WHERE post_count > 5"),

        # Multiple JOINs for reordering
        ("JOIN reordering", "SELECT u.name, p.title, c.content FROM users u JOIN posts p ON u.id = p.author_id JOIN comments c ON p.id = c.post_id WHERE u.active = 1"),

        # Complex WHERE with potential pushdown
        ("WHERE pushdown candidate", "SELECT u.name FROM users u JOIN posts p ON u.id = p.author_id WHERE u.age > 18 AND p.published = 1 AND u.status = 'active'"),

        # Redundant expressions
        ("Redundant expressions", "SELECT * FROM users WHERE age > 18 AND age > 18 AND status = 'active'"),

        # Boolean simplifications
        ("Boolean simplifications", "SELECT * FROM users WHERE NOT (age < 18) AND status = 'active'"),

        # Complex query with multiple optimizations
        ("Complex multi-optimization", "SELECT u.name, COUNT(p.id) FROM users u LEFT JOIN posts p ON u.id = p.author_id WHERE u.id = u.id AND EXISTS (SELECT 1 FROM comments c WHERE c.user_id = u.id) GROUP BY u.name HAVING COUNT(p.id) > 0"),
    ]

    success_count = 0
    total_tests = len(test_queries)

    for i, (description, query) in enumerate(test_queries, 1):
        print(f"Test {i}: {description}")
        print("Original query:")
        print(query)
        print()

        rewritten = rewriter.rewrite_query(query)

        print("Rewritten queries:")
        if rewritten:
            for j, q in enumerate(rewritten, 1):
                print(f"{j}. {q}")
            print("✅ Optimization applied")
            success_count += 1
        else:
            print("No rewrites generated")
            print("ℹ️  May be already optimal or no applicable transformations")
        print("-" * 50)

    print(f"\nResults: {success_count}/{total_tests} tests generated optimizations")
    return success_count > 0

if __name__ == "__main__":
    test_subquery_rewrite()