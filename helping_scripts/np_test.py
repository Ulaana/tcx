import numpy as np
from scripts.tcx_parser import parser_np

R = 6371000

def bbox_np(points_arr):
    min_lat, min_lon = np.min(points_arr[:, :2], axis=0)
    max_lat, max_lon = np.max(points_arr[:, :2], axis=0)
    return (min_lat, min_lon, max_lat, max_lon)

def total_distance_np(points_arr):
    rad_arr = np.radians(points_arr[:, :2])
    lat1, lon1 = rad_arr[:-1, 0], rad_arr[:-1, 1]
    lat2, lon2 = rad_arr[1:, 0], rad_arr[1:, 1]

    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1

    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * \
        np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = R * c
    return np.sum(distances)

def elevation_gain_np(points_arr):
    diffs = np.diff(points_arr[:, 2])
    return np.sum(diffs[diffs > 0])

def avg_hr_np(points_arr):
    hr_arr = points_arr[:, 3]
    valid_hr = hr_arr[hr_arr > 0]
    return np.mean(valid_hr) if valid_hr.size > 0 else 0

def hr_zones_np(points_arr, hr_max=185):
    hr_arr = points_arr[:, 3]
    bins = [hr_max * 0.5, hr_max * 0.6, hr_max * 0.7, hr_max * 0.8, hr_max * 0.9, np.inf]
    counts, _ = np.histogram(hr_arr, bins=bins)
    return counts.tolist()

def perf_eff_np(points, weight_kg=75.0):
    data = np.array(points)

    lat = np.radians(data[:, 0])
    lon = np.radians(data[:, 1])
    ele = data[:, 2]
    hr = data[:, 3]
    time = data[:, 4]

    dt = np.diff(time)
    dt = np.maximum(dt, 1.0)

    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    dist = R * c

    ele_diff = np.diff(ele)
    speed = dist / dt

    grade = np.divide(ele_diff, dist, out=np.zeros_like(ele_diff), where=(dist > 0))

    power = (weight_kg * 9.81 * speed * grade) + (0.5 * weight_kg * speed)
    power = np.maximum(0, power)

    window_size = 30
    window = np.ones(window_size) / window_size

    rolling_power = np.convolve(power, window, mode='valid')
    np_power = np.mean(rolling_power ** 4) ** 0.25 if len(rolling_power) > 0 else 0.0

    half_idx = len(power) // 2

    pow_1, pow_2 = power[:half_idx], power[half_idx:]
    hr_data = hr[1:]
    hr_1, hr_2 = hr_data[:half_idx], hr_data[half_idx:]

    avg_hr_1 = np.mean(hr_1)
    avg_hr_2 = np.mean(hr_2)

    ef1 = np.mean(pow_1) / avg_hr_1 if avg_hr_1 > 0 else 0.0
    ef2 = np.mean(pow_2) / avg_hr_2 if avg_hr_2 > 0 else 0.0

    decoupling = ((ef1 - ef2) / ef1 * 100) if ef1 > 0 else 0.0


    return {
        "Normalized_Power_W": round(float(np_power), 2),
        "Efficiency_Factor_H1": round(float(ef1), 3),
        "Efficiency_Factor_H2": round(float(ef2), 3),
        "Aerobic_Decoupling_Pct": round(float(decoupling), 2)
    }


if __name__ == "__main__":
    points_arr = parser_np("/mnt/d/personal/tcx/data/plik_1000000.tcx")
    bbox_np(points_arr)
    total_distance_np(points_arr)
    elevation_gain_np(points_arr)
    avg_hr_np(points_arr)
    hr_zones_np(points_arr, hr_max=185)
    perf_eff_np(points_arr)