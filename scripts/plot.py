from tcxreader.tcxreader import TCXReader
import numpy as np
import time
import matplotlib.pyplot as plt
from tcx_python import bbox, total_distance, elevation_gain, avg_hr, hr_zones, elevation_hr, perf_eff
from tcx_numpy import bbox_np, total_distance_np, elevation_gain_np, avg_hr_np, hr_zones_np, elevation_hr_np, perf_eff_np

def parser(tcx_file_path):
    tcx_reader = TCXReader()
    tcx = tcx_reader.read(tcx_file_path)
    points_list = []
    for trackpoint in tcx.trackpoints:
        lat = trackpoint.latitude if trackpoint.latitude else 0.0
        lon = trackpoint.longitude if trackpoint.longitude else 0.0
        if lat != 0.0 and lon != 0.0:
            ele = trackpoint.elevation if trackpoint.elevation else 0.0
            hr = trackpoint.hr_value if trackpoint.hr_value else 0.0
            cadence = trackpoint.cadence if trackpoint.cadence else 0.0
            points_list.append((lat, lon, ele, hr, cadence))
    points = len(points_list)
    points_arr = np.array(points_list)

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

    return points, time_p, time_n


def plot(files):
    results = []
    for file in files:
        points, time_py, time_np = parser(file)
        results.append((points, time_py, time_np))
    results.sort(key=lambda x: x[0])
    sizes = [r[0] for r in results]
    python_times = [r[1] for r in results]
    numpy_times = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, python_times, marker='o', linestyle='-', color='#e74c3c', linewidth=2.5, label='Python')
    plt.plot(sizes, numpy_times, marker='s', linestyle='-', color='#2980b9', linewidth=2.5, label='NumPy')
    plt.title('Porównanie czasu analizy na plikach TCX', fontsize=14, pad=15)
    plt.xlabel('Liczba punktów w pliku TCX', fontsize=12)
    plt.ylabel('Czas wykonania operacji (s)', fontsize=12)
    plt.xscale('log')
    plt.xticks(sizes, [f"{s:,}".replace(',', ' ') for s in sizes])
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()

    plt.savefig('../wykres.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    files = [
        "../data/plik_1000.tcx",
        "../data/plik_10000.tcx",
        "../data/plik_100000.tcx",
        "../data/plik_500000.tcx",
        "../data/plik_1000000.tcx"
    ]
    plot(files)
