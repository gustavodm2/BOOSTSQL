#!/usr/bin/env python3

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_status():
    print("\nTesting status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Knowledge base patterns: {data['knowledge_base_patterns']}")
            print(f"Optimizations learned: {data['optimizations_learned']}")
            print(f"Strategies available: {data['strategies_available']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_transforms():
    print("\nTesting transforms endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/transforms")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available transformations: {data['transformations']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_rewrite():
    print("\nTesting rewrite endpoint...")
    query = "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)"

    try:
        response = requests.post(
            f"{BASE_URL}/rewrite",
            json={"query": query, "transformation": "subquery_to_join"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Original: {data['original_query'][:50]}...")
            if data['transformations']:
                for transform, queries in data['transformations'].items():
                    print(f"{transform}: {len(queries)} rewritten versions")
                    if queries:
                        print(f"  Example: {queries[0][:50]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_optimize():
    print("\nTesting optimize endpoint...")
    query = "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)"

    try:
        response = requests.post(
            f"{BASE_URL}/optimize",
            json={"query": query}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Candidates evaluated: {data['all_candidates_evaluated']}")
            print(f"Successful optimizations: {data['successful_optimizations']}")
            if data['best_optimization']:
                print(f"Best optimization: {data['best_optimization']['optimization_type']}")
                print(f"Improvement ratio: {data['best_optimization']['improvement_ratio']:.2f}x")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("SQLBoost API Test Suite")
    print("=" * 50)

    print("Waiting for API to start...")
    time.sleep(2)

    tests = [
        ("Health Check", test_health),
        ("Status", test_status),
        ("Transforms", test_transforms),
        ("Rewrite", test_rewrite),
        ("Optimize", test_optimize),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API logs for details.")

if __name__ == "__main__":
    main()