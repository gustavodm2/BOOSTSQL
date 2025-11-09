#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.query_rewriter import SQLQueryRewriter

def test_subquery_rewrite():
    rewriter = SQLQueryRewriter()

    # Test query with subquery
    query = "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)"

    print("Original query:")
    print(query)
    print()

    # Test subquery to join transformation
    rewritten = rewriter.rewrite_query(query, 'subquery_to_join')

    print("Rewritten queries:")
    for i, q in enumerate(rewritten, 1):
        print(f"{i}. {q}")
    print()

    # Check if it's different
    if len(rewritten) > 0 and rewritten[0] != query:
        print("✅ Query rewriting is working!")
        return True
    else:
        print("❌ Query rewriting is not working")
        return False

if __name__ == "__main__":
    test_subquery_rewrite()