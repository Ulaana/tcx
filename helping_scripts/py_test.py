import math
from tcxreader.tcxreader import TCXReader

R = 6371000

def bbox(points):
    min_lat, min_lon = float('inf'), float('inf')
    max_lat, max_lon = float('-inf'), float('-inf')
    for p in points:
        lat, lon = p[0], p[1]
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

def total_distance(points):
    total_distance = 0.0
    for i in range(1, len(points)):
        lat1, lon1 = points[i - 1][0], points[i - 1][1]
        lat2, lon2 = points[i][0], points[i][1]
        total_distance += distance(lat1, lon1, lat2, lon2)
    return total_distance

def elevation_gain(points):
    gain = 0.0
    for i in range(1, len(points)):
        diff = points[i][2] - points[i - 1][2]
        if diff > 0:
            gain += diff
    return gain

def avg_hr(points):
    total_hr = 0
    count = 0
    for p in points:
        hr = p[3]
        if hr > 0:
            total_hr += hr
            count += 1
    return total_hr / count if count > 0 else 0

def hr_zones(points, hr_max=185):
    z1_min, z2_min, z3_min, z4_min, z5_min = [hr_max * p for p in (0.5, 0.6, 0.7, 0.8, 0.9)]
    zones = [0, 0, 0, 0, 0]
    for p in points:
        hr = p[3]
        if hr >= z5_min: zones[4] += 1
        elif hr >= z4_min: zones[3] += 1
        elif hr >= z3_min: zones[2] += 1
        elif hr >= z2_min: zones[1] += 1
        elif hr >= z1_min: zones[0] += 1
    return zones

def perf_eff(points, weight_kg=75.0):

    delta_times = []
    distances = []
    speeds = []
    grades = []
    powers = []

    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]

        dt = p2[4] - p1[4]
        if dt <= 0:
            dt = 1.0

        dist = distance(p1[0], p1[1], p2[0], p2[1])
        ele_diff = p2[2] - p1[2]
        speed = dist / dt

        grade = (ele_diff / dist) if dist > 0 else 0

        power = (weight_kg * 9.81 * speed * grade) + (0.5 * weight_kg * speed)
        power = max(0, power)

        delta_times.append(dt)
        distances.append(dist)
        speeds.append(speed)
        grades.append(grade)
        powers.append(power)

    rolling_powers_4th = []
    window_size = 30

    for i in range(len(powers)):
        start_idx = max(0, i - window_size + 1)
        window = powers[start_idx:i + 1]
        rolling_avg = sum(window) / len(window)
        rolling_powers_4th.append(rolling_avg ** 4)

    np_power = (sum(rolling_powers_4th) / len(rolling_powers_4th)) ** 0.25 if rolling_powers_4th else 0

    half_idx = len(powers) // 2

    def calc_efficiency_factor(pow_list, hr_list):
        avg_p = sum(pow_list) / len(pow_list) if pow_list else 0
        avg_hr = sum(hr_list) / len(hr_list) if hr_list else 1
        return avg_p / avg_hr

    hr_data = [p[3] for p in points[1:]]

    ef_first_half = calc_efficiency_factor(powers[:half_idx], hr_data[:half_idx])
    ef_second_half = calc_efficiency_factor(powers[half_idx:], hr_data[half_idx:])

    decoupling = ((ef_first_half - ef_second_half) / ef_first_half * 100) if ef_first_half > 0 else 0


    return {
        "Normalized_Power_W": round(np_power, 2),
        "Efficiency_Factor_H1": round(ef_first_half, 3),
        "Efficiency_Factor_H2": round(ef_second_half, 3),
        "Aerobic_Decoupling_Pct": round(decoupling, 2)
    }

if __name__ == "__main__":
    tcx_reader = TCXReader()
    tcx = tcx_reader.read("/mnt/d/personal/tcx/data/plik_10000000.tcx")

    points_list = []
    for trackpoint in tcx.trackpoints:
        lat = trackpoint.latitude if trackpoint.latitude else 0.0
        lon = trackpoint.longitude if trackpoint.longitude else 0.0
        if lat != 0.0 and lon != 0.0:
            ele = trackpoint.elevation if trackpoint.elevation else 0.0
            hr = trackpoint.hr_value if trackpoint.hr_value else 0.0
            cadence = trackpoint.cadence if trackpoint.cadence else 0.0
            points_list.append((lat, lon, ele, hr, cadence))

    bbox(points_list)
    total_distance(points_list)
    elevation_gain(points_list)
    avg_hr(points_list)
    hr_zones(points_list, hr_max=185)
    perf_eff((points_list))