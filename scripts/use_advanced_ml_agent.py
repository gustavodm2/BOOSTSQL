import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def main():
    print('SQLBOOST - ADVANCED ML AGENT OPTIMIZER')
    print('An intelligent agent that learns and optimizes SQL queries')
    print()

    try:
        from src.ml_agent import SQLBoostMLAgent

        db_config = config.get_database_config()

        agent = SQLBoostMLAgent(db_config)

        print(' Advanced ML Agent loaded!')
        print('Agent Status:')
        status = agent.get_agent_status()
        print(f'  Knowledge base: {status["knowledge_base_patterns"]} patterns')
        print(f'  Learned optimizations: {status["optimizations_learned"]}')
        print(f'  Available strategies: {status["strategies_available"]}')
        print()

        print(' Enter SQL queries for intelligent optimization analysis:')
        print("   (type 'status' to see agent status, 'rewrite <query>' for direct rewriting, 'quit' to exit)")
        print()

        while True:
            try:
                query = input('SQL Query: ').strip()
                if not query:
                    continue

                if query.lower() in ['quit', 'exit', 'sair']:
                    print(' Goodbye!')
                    break

                if query.lower() == 'status':
                    status = agent.get_agent_status()
                    print('\n Agent Status:')
                    print(f'  Knowledge base patterns: {status["knowledge_base_patterns"]}')
                    print(f'  Optimizations learned: {status["optimizations_learned"]}')
                    print(f'  Exploration rate: {status["exploration_rate"]:.3f}')
                    print(f'  Learning rate: {status["learning_rate"]:.3f}')
                    print('\n Strategy Performance:')
                    for strategy_name, metrics in status['strategy_effectiveness'].items():
                        print(f'  {strategy_name}: {metrics["success_rate"]:.1%} success, {metrics["avg_improvement"]:.2f}x avg improvement')
                    print()
                    continue

                if query.lower().startswith('rewrite '):
                    actual_query = query[8:].strip()
                    if not actual_query:
                        print(' Please provide a query to rewrite. Usage: rewrite <sql_query>')
                        continue

                    print(f'\n Rewriting query: {actual_query[:100]}...' if len(actual_query) > 100 else f'\n Rewriting query: {actual_query}')
                    print('\n REWRITTEN VERSIONS:')

                    transformations = ['subquery_to_join', 'cte_materialization', 'simplify_expressions', 'join_reordering']
                    any_rewrites = False

                    for transformation in transformations:
                        try:
                            rewritten = agent.query_rewriter.rewrite_query(actual_query, transformation)
                            if len(rewritten) > 1 or (len(rewritten) == 1 and rewritten[0] != actual_query):
                                any_rewrites = True
                                print(f'\n {transformation.replace("_", " ").title()}:')
                                for i, rewritten_query in enumerate(rewritten, 1):
                                    if rewritten_query != actual_query:
                                        print(f'   {i}. {rewritten_query}')
                        except Exception as e:
                            print(f'    {transformation}: Failed - {e}')

                    if not any_rewrites:
                        print('   No transformations could be applied to this query.')

                    print('-' * 80)
                    continue

                print('\n Analyzing query...')
                analysis = agent.optimize_query(query)

                print(' ANALYSIS RESULTS:')
                print(f'Original execution time: {analysis["original_execution_time"]:.2f}ms')

                if analysis['best_optimization']:
                    best = analysis['best_optimization']
                    print(f'Best optimization: {best["optimization_type"]}')
                    print(f'Improvement ratio: {best["improvement_ratio"]:.2f}x')
                    print(f'Optimized time: {best["optimized_time"]:.2f}ms')
                    if len(best['optimized_query']) < 200:
                        print(f'Optimized query: {best["optimized_query"]}')
                    else:
                        print(f'Optimized query: {best["optimized_query"][:200]}...')

                print(f'Candidates evaluated: {analysis["all_candidates_evaluated"]}')
                print(f'Successful optimizations: {analysis["successful_optimizations"]}')

                insights = analysis['agent_insights']
                print(f'Complexity: {insights["complexity_assessment"]}')
                print(f'Performance prediction: {insights["performance_prediction"]}')
                print(f'Similar queries analyzed: {insights["similar_queries_analyzed"]}')

                if analysis['recommendations']:
                    print('\n RECOMMENDATIONS:')
                    for rec in analysis['recommendations']:
                        print(f'   • {rec}')

                if insights['potential_bottlenecks']:
                    print('\n  POTENTIAL BOTTLENECKS:')
                    for bottleneck in insights['potential_bottlenecks']:
                        print(f'   • {bottleneck.replace("_", " ").title()}')

                print('-' * 80)

            except KeyboardInterrupt:
                print('\n Goodbye!')
                break
            except Exception as e:
                print(f' Error: {e}')
                print('Please try again or check your query syntax.')

    except ImportError as e:
        print(f' Missing dependencies: {e}')
        print('Install required packages: pip install -r requirements.txt')
    except Exception as e:
        print(f' Failed to initialize agent: {e}')
        print('Make sure you have trained the agent first: python scripts/train_advanced_ml_agent.py')

if __name__ == '__main__':
    main()