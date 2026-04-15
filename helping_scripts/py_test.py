import math
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from scripts.tcx_parser import parser_py

R = 6371000


def bbox(latitudes, longitudes):
    min_lat, min_lon = float('inf'), float('inf')
    max_lat, max_lon = float('-inf'), float('-inf')

    for lat, lon in zip(latitudes, longitudes):
        if lat < min_lat: min_lat = lat
        if lat > max_lat: max_lat = lat
        if lon < min_lon: min_lon = lon
        if lon > max_lon: max_lon = lon

    return (min_lat, min_lon, max_lat, max_lon)

def distance(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def total_distance(latitudes, longitudes):
    total_distance = 0.0
    for i in range(1, len(latitudes)):
        total_distance += distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
    return total_distance


def elevation_gain(elevations):
    gain = 0.0
    for i in range(1, len(elevations)):
        diff = elevations[i] - elevations[i - 1]
        if diff > 0:
            gain += diff
    return gain


def avg_hr(heart_rates):
    total_hr = 0
    count = 0
    for hr in heart_rates:
        if hr > 0:
            total_hr += hr
            count += 1
    return total_hr / count if count > 0 else 0


def hr_zones(heart_rates, hr_max=185):
    z1_min, z2_min, z3_min, z4_min, z5_min = [hr_max * p for p in (0.5, 0.6, 0.7, 0.8, 0.9)]
    zones = [0, 0, 0, 0, 0]
    for hr in heart_rates:
        if hr >= z5_min:
            zones[4] += 1
        elif hr >= z4_min:
            zones[3] += 1
        elif hr >= z3_min:
            zones[2] += 1
        elif hr >= z2_min:
            zones[1] += 1
        elif hr >= z1_min:
            zones[0] += 1
    return zones


def elevation_hr(elevations, heart_rates):
    climb_hr_total = 0
    climb_count = 0
    descent_hr_total = 0
    descent_count = 0
    for i in range(1, len(elevations)):
        ele_diff = elevations[i] - elevations[i - 1]
        hr = heart_rates[i]
        if hr > 0:
            if ele_diff > 0:
                climb_hr_total += hr
                climb_count += 1
            else:
                descent_hr_total += hr
                descent_count += 1

    avg_climb = climb_hr_total / climb_count if climb_count > 0 else 0
    avg_descent = descent_hr_total / descent_count if descent_count > 0 else 0
    return avg_climb, avg_descent


def perf_eff(latitudes, longitudes, elevations, heart_rates, times, weight=75.0):
    delta_times = []
    distances = []
    speeds = []
    grades = []
    powers = []

    for i in range(1, len(latitudes)):
        delta_time = times[i] - times[i - 1]
        if delta_time <= 0:
            delta_time = 1.0

        dist = distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
        ele_diff = elevations[i] - elevations[i - 1]
        speed = dist / delta_time
        grade = (ele_diff / dist) if dist > 0 else 0
        power = (weight * 9.81 * speed * grade) + (0.5 * weight * speed)
        power = max(0, power)

        delta_times.append(delta_time)
        distances.append(dist)
        speeds.append(speed)
        grades.append(grade)
        powers.append(power)

    power_4th = []
    window_size = 30
    for i in range(len(powers)):
        start = max(0, i - window_size + 1)
        window = powers[start:i + 1]
        rolling_mean = sum(window) / len(window)
        power_4th.append(rolling_mean ** 4)

    normalized_power = (sum(power_4th) / len(power_4th)) ** 0.25 if power_4th else 0
    half = len(powers) // 2

    def calculate_ef(power, hr):
        avg_p = sum(power) / len(power) if power else 0
        avg_hr = sum(hr) / len(hr) if hr else 1
        return avg_p / avg_hr

    hr_data = heart_rates[1:]

    ef_1 = calculate_ef(powers[:half], hr_data[:half])
    ef_2 = calculate_ef(powers[half:], hr_data[half:])
    decoupling = ((ef_1 - ef_2) / ef_1 * 100) if ef_2 > 0 else 0

    return {
        "Znormalizowana moc (W)": round(normalized_power, 2),
        "Współczynnik wydajności (pierwsza połowa treningu)": round(ef_1, 3),
        "Współczynnik wydajności (druga połowa treningu)": round(ef_2, 3),
        "Aerobic decoupling": round(decoupling, 2)
    }


if __name__ == "__main__":
    i = 3
    latitudes, longitudes, elevations, heart_rates, times, cadences = parser_py("/mnt/d/personal/tcx/data/plik_1000000.tcx")
    for _ in range(i):
        bbox(latitudes, longitudes)
        total_distance(latitudes, longitudes)
        elevation_gain(elevations)
        avg_hr(heart_rates)
        hr_zones(heart_rates, hr_max=185)
        elevation_hr(elevations, heart_rates)
        perf_eff(latitudes, longitudes, elevations, heart_rates, times)