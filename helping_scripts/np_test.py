import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from scripts.tcx_numpy import *
from scripts.tcx_parser import parser_np


if __name__ == "__main__":
    i = 0
    latitudes, longitudes, elevations, heart_rates, times, cadences = parser_np("/mnt/d/personal/tcx/data/plik_100000.tcx")
    for _ in range(i):
        total_distance_np(latitudes, longitudes)
        elevation_gain_np(elevations)
        avg_hr_np(heart_rates)
        hr_zones_np(heart_rates, hr_max=185)
        elevation_hr_np(elevations, heart_rates)
        perf_eff_np(latitudes, longitudes, elevations, heart_rates, times)
        activity_segments_np(latitudes, longitudes, times, cadences)

