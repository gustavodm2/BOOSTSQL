import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.query_optimizer import SQLBoostOptimizer

def main():
    print('SQLBOOST - OTIMIZADOR DE QUERIES')
    model_path = 'models/sqlboost_500k.pkl'
    if not os.path.exists(model_path):
        print(f'Modelo não encontrado: {model_path}')
        print('Execute: python scripts/train_model.py')
        return
    optimizer = SQLBoostOptimizer(model_path=model_path)
    print('Modelo carregado')
    print('\nDigite queries SQL para análise:')
    print("   (digite 'quit' para sair)\n")
    while True:
        try:
            query = input('SQL: ').strip()
            if query.lower() in ['quit', 'exit', 'sair']:
                break
            if not query:
                continue
            analysis = optimizer.suggest_optimizations(query)
            print(f'\nANÁLISE:')
            print(f'Tempo previsto: {analysis['predicted_execution_time_ms']:.2f} ms')
            print(f'Complexidade: {analysis['query_complexity']:.2f}/1.0')
            if analysis['suggestions']:
                print('SUGESTÕES:')
                for suggestion in analysis['suggestions']:
                    print(f'   • {suggestion}')
            else:
                print('Query bem otimizada!')
            print('-' * 50)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f'Erro: {e}')
if __name__ == '__main__':
    main()
