import csv

def calculate_average_improvement(csv_file='benchmark_results.csv', min_orig_time=1.0, min_improvement=5.0):

    improvements = []
    total_queries = 0
    improved_queries = 0

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_queries += 1
                try:
                    orig = float(row['original_execution_time'])
                    opt = float(row['optimized_execution_time'])
                    if orig > 0:
                        imp = ((orig - opt) / orig) * 100
                        improvements.append(imp)
                        if opt < orig and imp >= min_improvement:
                            improved_queries += 1
                except (ValueError, KeyError):
                    continue  

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"Média de melhoria: {avg_improvement:.2f}%")
            return avg_improvement
        else:
            print("Nenhuma melhoria válida encontrada.")
            return None
    except FileNotFoundError:
        print(f"Arquivo {csv_file} não encontrado.")
        return None

if __name__ == "__main__":
    calculate_average_improvement()