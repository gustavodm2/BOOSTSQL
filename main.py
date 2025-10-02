import os
import pandas as pd
from src.query_generator import MassiveQueryGenerator
from src.query_optimizer import SQLBoostOptimizer

def main():
    print("🚀 INICIANDO SQLBOOST - OTIMIZADOR INTELIGENTE DE QUERIES")
    
    try:
        print("\nGERANDO DATASET DE TREINAMENTO...")
        generator = MassiveQueryGenerator()
        dataset = generator.generate_dataset(num_queries=1000)  
        
        print(f"Geradas {len(dataset)} queries para treinamento")
        
        print("\nTREINANDO MODELO DE MACHINE LEARNING...")
        optimizer = SQLBoostOptimizer()
        
        results = optimizer.train(dataset, 'sqlboost_model.pkl')
        
        print("\nTESTANDO O MODELO...")
        
        test_queries = [
            "SELECT name, email FROM users WHERE status = 'active'",
            
            """SELECT u.name, COUNT(o.id) as order_count 
               FROM users u 
               JOIN orders o ON u.id = o.user_id 
               WHERE u.created_at > '2023-01-01' 
               GROUP BY u.name 
               HAVING COUNT(o.id) > 5""",
            
            """SELECT c.name, p.category, SUM(o.quantity * p.price) as total_sales
               FROM customers c
               JOIN orders o ON c.id = o.customer_id
               JOIN products p ON o.product_id = p.id
               WHERE o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
               AND p.price > 100
               GROUP BY c.name, p.category
               ORDER BY total_sales DESC
               LIMIT 10"""
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- QUERY DE TESTE {i} ---")
            print(f"SQL: {query[:100]}..." if len(query) > 100 else f"SQL: {query}")
            
            analysis = optimizer.suggest_optimizations(query)
            
            print(f"Tempo previsto: {analysis['predicted_execution_time_ms']:.2f} ms")
            print(f"Complexidade: {analysis['query_complexity']:.2f}/1.0")
            
            if analysis['suggestions']:
                print("Sugestões de otimização:")
                for suggestion in analysis['suggestions']:
                    print(f"   • {suggestion}")
            else:
                print("Query bem otimizada!")
        
        print(f"\nSQLBOOST CONFIGURADO COM SUCESSO!")
        print(f"Modelo treinado com {len(dataset)} queries")
        print(f"Pronto para otimizar suas queries SQL!")
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        print("Dica: Verifique se todos os arquivos estão na pasta src/")

if __name__ == "__main__":
    main()