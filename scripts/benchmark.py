from tcxreader.tcxreader import TCXReader
import numpy as np
import time

from tcx_python import bbox, total_distance, elevation_gain, avg_hr, hr_zones, elevation_hr, perf_eff
from tcx_numpy import bbox_np, total_distance_np, elevation_gain_np, avg_hr_np, hr_zones_np, elevation_hr_np, \
    perf_eff_np


def benchmark(tcx_file_path):
    print(f"Wczytywanie pliku TCX: {tcx_file_path}")
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

    print(f"Liczba punktów do analizy: {len(points_list)}\n")

    points_arr = np.array(points_list)

    start = time.perf_counter()
    bbox_p = bbox(points_list)
    time_bbox_p = time.perf_counter() - start

    start = time.perf_counter()
    bbox_n = bbox_np(points_arr)
    time_bbox_n = time.perf_counter() - start

    start = time.perf_counter()
    dist_p = total_distance(points_list)
    time_dist_p = time.perf_counter() - start

    start = time.perf_counter()
    dist_n = total_distance_np(points_arr)
    time_dist_n = time.perf_counter() - start

    start = time.perf_counter()
    ele_gain_p = elevation_gain(points_list)
    time_ele_gain_p = time.perf_counter() - start

    start = time.perf_counter()
    ele_gain_n = elevation_gain_np(points_arr)
    time_ele_gain_n = time.perf_counter() - start

    start = time.perf_counter()
    hr_p = avg_hr(points_list)
    time_hr_p = time.perf_counter() - start

    start = time.perf_counter()
    hr_n = avg_hr_np(points_arr)
    time_hr_n = time.perf_counter() - start

    start = time.perf_counter()
    zones_p = hr_zones(points_list, hr_max=185)
    time_zones_p = time.perf_counter() - start

    start = time.perf_counter()
    zones_n = hr_zones_np(points_arr, hr_max=185)
    time_zones_n = time.perf_counter() - start

    start = time.perf_counter()
    ele_hr_p = elevation_hr(points_list)
    time_ele_hr_p = time.perf_counter() - start

    start = time.perf_counter()
    ele_hr_n = elevation_hr_np(points_arr)
    time_ele_hr_n = time.perf_counter() - start

    start = time.perf_counter()
    perf_eff_p = perf_eff(points_list)
    time_perf_eff_p = time.perf_counter() - start

    start = time.perf_counter()
    perf_eff_n = perf_eff_np(points_arr)
    time_perf_eff_n = time.perf_counter() - start

    print("Wyniki analizy:")
    print(f"Dystans: {dist_n / 1000:.2f} km | Przewyższenia: {ele_gain_n:.0f} m | Średnie tętno: {hr_n:.0f} BPM")
    print(f"Czas w strefach HR (Z1-Z5): {zones_n}\n")

    print("| Operacja | Czas Python (s) | Czas NumPy (s) | Przyspieszenie (Py/Np) |")
    print(f"| Bounding Box | {time_bbox_p:.6f} | {time_bbox_n:.6f} | {time_bbox_p / time_bbox_n:.2f}x |")
    print(f"| Dystans | {time_dist_p:.6f} | {time_dist_n:.6f} | {time_dist_p / time_dist_n:.2f}x |")
    print(f"| Przewyższenia | {time_ele_gain_p:.6f} | {time_ele_gain_n:.6f} | {time_ele_gain_p / time_ele_gain_n:.2f}x |")
    print(f"| Średnie tętno | {time_hr_p:.6f} | {time_hr_n:.6f} | {time_hr_p / time_hr_n:.2f}x |")
    print(f"| HR Zones | {time_zones_p:.6f} | {time_zones_n:.6f} | {time_zones_p / time_zones_n:.2f}x |")
    print(f"| Średnie tętno na podjazdach | {time_ele_hr_p:.6f} | {time_ele_hr_n:.6f} | {time_ele_hr_p / time_ele_hr_n:.2f}x |")
    print(f"| Perfomance Efficiency | {time_perf_eff_p:.6f} | {time_perf_eff_n:.6f} | {time_perf_eff_p / time_perf_eff_n:.2f}x |")


if __name__ == "__main__":
    benchmark("../data/plik_1000000.tcx")
