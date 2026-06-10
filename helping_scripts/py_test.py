import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from scripts.tcx_python import *
from scripts.tcx_parser import parser_py


if __name__ == "__main__":
    i = 5
    latitudes, longitudes, elevations, heart_rates, times, cadences = parser_py("/mnt/d/personal/tcx/data/plik_100000.tcx")
    for _ in range(i):
        total_distance(latitudes, longitudes)
        elevation_gain(elevations)
        avg_hr(heart_rates)
        hr_zones(heart_rates, hr_max=185)
        elevation_hr(elevations, heart_rates)
        perf_eff(latitudes, longitudes, elevations, heart_rates, times)
        activity_segments(latitudes, longitudes, times, cadences)