import math

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

def elevation_hr(points):
    climb_hr_total = 0
    climb_count = 0
    descent_hr_total = 0
    descent_count = 0
    for i in range(1, len(points)):
        ele_diff = points[i][2] - points[i - 1][2]
        hr = points[i][3]
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