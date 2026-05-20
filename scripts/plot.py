import json
import matplotlib.pyplot as plt


def plot(file):
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    results.sort(key=lambda x: x["punkty"])

    sizes = [r["punkty"] for r in results]
    #przyspieszenie = [r["przyspieszenie"] for r in results]
    python_times = [r["czas_python"] for r in results]
    numpy_times = [r["czas_numpy"] for r in results]

    plt.figure(figsize=(10, 6))
    #plt.plot(sizes, przyspieszenie, marker='o', linestyle='-', color='#e74c3c', linewidth=2.5, label='Przyspieszenie')
    plt.plot(sizes, python_times, marker='o', linestyle='-', color='#e74c3c', linewidth=2.5, label='Python')
    plt.plot(sizes, numpy_times, marker='s', linestyle='-', color='#2980b9', linewidth=2.5, label='NumPy')
    plt.title('Porównanie czasu wykonania między NumPy a Pythonem na plikach TCX', fontsize=14, pad=15)
    plt.xlabel('Liczba punktów w pliku TCX', fontsize=12)
    plt.ylabel('Czas wykonania operacji analitycznych (s)', fontsize=12)
    #plt.ylabel('Przyspieszenie', fontsize=12)
    plt.xscale('log')
    plt.xticks(sizes, [f"{s:,}".replace(',', ' ') for s in sizes])
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()

    plt.savefig('../results/wykres.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot("../results/benchmark.json")
