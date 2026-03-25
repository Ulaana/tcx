import time
import json

from tcx_python import bbox, total_distance, elevation_gain, avg_hr, hr_zones, elevation_hr, perf_eff
from tcx_numpy import bbox_np, total_distance_np, elevation_gain_np, avg_hr_np, hr_zones_np, elevation_hr_np, \
    perf_eff_np
from tcx_parser import parser_py, parser_np


def benchmark(files, output_file="../benchmark.json"):
    results = []

    for file in files:
        print(f"Przetwarzanie pliku: {file}...")
        points_list = parser_py(file)
        points = len(points_list)
        points_arr = parser_np(file)

        start_p = time.perf_counter()
        bbox(points_list)
        total_distance(points_list)
        elevation_gain(points_list)
        avg_hr(points_list)
        hr_zones(points_list)
        elevation_hr(points_list)
        perf_eff(points_list)
        time_p = time.perf_counter() - start_p

        start_n = time.perf_counter()
        bbox_np(points_arr)
        total_distance_np(points_arr)
        elevation_gain_np(points_arr)
        avg_hr_np(points_arr)
        hr_zones_np(points_arr)
        elevation_hr_np(points_arr)
        perf_eff_np(points_arr)
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