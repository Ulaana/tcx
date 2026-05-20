import time
import json

from tcx_python import *
from tcx_numpy import *
from tcx_parser import *

def benchmark_functions(file, output_file="../results/benchmark_functions.json", iterations=100):
    print(f"Wczytywanie i parsowanie pliku: {file}...")

    latitudes, longitudes, elevations, heart_rates, times, cadences = parser_py(file)
    latitudes_np, longitudes_np, elevations_np, heart_rates_np, times_np, cadences_np = parser_np(file)
    points = len(latitudes)

    results = []

    def measure_pair(func_name, func_py, args_py, func_np, args_np):
        start_p = time.perf_counter()
        for _ in range(iterations):
            func_py(*args_py)
        time_p = (time.perf_counter() - start_p) / iterations

        start_n = time.perf_counter()
        for _ in range(iterations):
            func_np(*args_np)
        time_n = (time.perf_counter() - start_n) / iterations

        speedup = time_p / time_n if time_n > 0 else 0

        print(f"{func_name:<20} | Python: {time_p:.6f}s | NumPy: {time_n:.6f}s | Przyspieszenie: {speedup:.2f}x")

        results.append({
            "funkcja": func_name,
            "czas_python_s": time_p,
            "czas_numpy_s": time_n,
            "przyspieszenie": speedup
        })

    print(f"\nRozpoczynanie benchmarku dla {points} punktów (iteracje dla każdej funkcji: {iterations})\n" + "-" * 75)

    measure_pair("total_distance",
                 total_distance, (latitudes, longitudes),
                 total_distance_np, (latitudes_np, longitudes_np))

    measure_pair("elevation_gain",
                 elevation_gain, (elevations,),
                 elevation_gain_np, (elevations_np,))

    measure_pair("avg_hr",
                 avg_hr, (heart_rates,),
                 avg_hr_np, (heart_rates_np,))

    measure_pair("hr_zones",
                 hr_zones, (heart_rates,),
                 hr_zones_np, (heart_rates_np,))

    measure_pair("elevation_hr",
                 elevation_hr, (elevations, heart_rates),
                 elevation_hr_np, (elevations_np, heart_rates_np))

    measure_pair("perf_eff",
                 perf_eff, (latitudes, longitudes, elevations, heart_rates, times),
                 perf_eff_np, (latitudes_np, longitudes_np, elevations_np, heart_rates_np, times_np))

    measure_pair("activity_segments",
                 activity_segments, (latitudes, longitudes, times, cadences),
                 activity_segments_np, (latitudes_np, longitudes_np, times_np, cadences_np))

    final_output = {
        "plik": file,
        "punkty": points,
        "iteracje_pomiarowe": iterations,
        "wyniki": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    print(f"\nGotowe! Zapisano szczegółowe wyniki do pliku: {output_file}")

if __name__ == "__main__":
    benchmark_functions("../data/plik_1000000.tcx", iterations=50)