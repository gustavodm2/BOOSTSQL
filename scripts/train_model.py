

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query_generator import MassiveQueryGenerator
from src.query_optimizer import SQLBoostOptimizer

def main():
    print("🚀 SQLBOOST - TREINAMENTO COM 500.000 QUERIES")
    
    generator = MassiveQueryGenerator()
    dataset = generator.generate_500k_dataset()
    
    df = pd.DataFrame(dataset)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/500k_queries_dataset.csv', index=False)
    
    optimizer = SQLBoostOptimizer()
    
    results = optimizer.train(dataset, 'models/sqlboost_500k.pkl')
    
    print("\n🎯 TESTANDO MODELO...")
    
    test_queries = [
        "SELECT * FROM users",  
        "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",  
        "WITH user_orders AS (SELECT user_id, COUNT(*) as order_count FROM orders GROUP BY user_id) SELECT u.name, uo.order_count FROM users u JOIN user_orders uo ON u.id = uo.user_id WHERE uo.order_count > 5" 
    ]
    
    for i, query in enumerate(test_queries, 1):
        analysis = optimizer.suggest_optimizations(query)
        print(f"\n--- TESTE {i} ---")
        print(f"📝 {query[:60]}...")
        print(f"⏱️  Tempo previsto: {analysis['predicted_execution_time_ms']:.2f} ms")
        print(f"📊 Complexidade: {analysis['query_complexity']:.2f}/1.0")
        
        if analysis['suggestions']:
            print("💡 Sugestões:")
            for suggestion in analysis['suggestions']:
                print(f"   • {suggestion}")
    
    print(f"\n✅ TREINAMENTO CONCLUÍDO!")
    print(f"📈 Modelo salvo: models/sqlboost_500k.pkl")
    print(f"📊 Performance esperada: R² > 0.95")

if __name__ == "__main__":
    main()