import time
import json

from tcx_python import *
from tcx_numpy import *
from tcx_parser import *


def benchmark(files, output_file="../benchmark.json"):
    results = []

    for file in files:
        print(f"Przetwarzanie pliku: {file}...")
        latitudes, longitudes, elevations, heart_rates, times, cadences = parser_py(file)
        points = len(latitudes)
        latitudes_np, longitudes_np, elevations_np, heart_rates_np, times_np, cadences_np = parser_np(file)

        start_p = time.perf_counter()
        bbox(latitudes, longitudes)
        total_distance(latitudes, longitudes)
        elevation_gain(elevations)
        avg_hr(heart_rates)
        hr_zones(heart_rates)
        elevation_hr(elevations, heart_rates)
        perf_eff(latitudes, longitudes, elevations, heart_rates, times)
        time_p = time.perf_counter() - start_p

        start_n = time.perf_counter()
        bbox_np(latitudes_np, longitudes_np)
        total_distance_np(latitudes_np, longitudes_np)
        elevation_gain_np(elevations_np)
        avg_hr_np(heart_rates_np)
        hr_zones_np(heart_rates_np)
        elevation_hr_np(elevations_np, heart_rates_np)
        perf_eff_np(latitudes_np, longitudes_np, elevations_np, heart_rates_np, times_np)
        time_n = time.perf_counter() - start_n

        results.append({
            "plik": file,
            "punkty": points,
            "czas_python": time_p,
            "czas_numpy": time_n,
            "przyspieszenie": time_p/time_n
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\nGotowe! Zapisano wyniki do pliku: {output_file}")


if __name__ == "__main__":
    files = [
        "../data/plik_1000.tcx",
        "../data/plik_10000.tcx",
        "../data/plik_100000.tcx",
        "../data/plik_500000.tcx",
        "../data/plik_1000000.tcx"
    ]
    benchmark(files)