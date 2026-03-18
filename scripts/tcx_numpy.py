import numpy as np

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


def elevation_hr_np(points_arr):
    ele_diffs = np.diff(points_arr[:, 2])
    hr_segments = points_arr[1:, 3]
    valid_hr = hr_segments > 0
    is_climb = ele_diffs > 0
    is_descent_or_flat = ele_diffs <= 0
    hr_on_climbs = hr_segments[valid_hr & is_climb]
    hr_on_descents = hr_segments[valid_hr & is_descent_or_flat]
    avg_climb = np.mean(hr_on_climbs) if hr_on_climbs.size > 0 else 0
    avg_descent = np.mean(hr_on_descents) if hr_on_descents.size > 0 else 0
    return avg_climb, avg_descent


def perf_eff_np(points_arr, weight=75.0):
    lat = np.radians(points_arr[:, 0])
    lon = np.radians(points_arr[:, 1])
    ele = points_arr[:, 2]
    hr = points_arr[:, 3]
    time = points_arr[:, 4]
    delta_time = np.diff(time)
    delta_time = np.maximum(delta_time, 1.0)
    delta_lat = np.diff(lat)
    delta_lon = np.diff(lon)
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(delta_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    dist = R * c
    ele_diff = np.diff(ele)
    speed = dist / delta_time
    grade = np.divide(ele_diff, dist, out=np.zeros_like(ele_diff), where=(dist > 0))
    power = (weight * 9.81 * speed * grade) + (0.5 * weight * speed)
    power = np.maximum(0, power)

    window_size = 30
    window = np.ones(window_size) / window_size
    rolling_mean = np.convolve(power, window, mode='valid')
    normalized_power = np.mean(rolling_mean ** 4) ** 0.25 if len(rolling_mean) > 0 else 0.0
    half = len(power) // 2
    pow_1, pow_2 = power[:half], power[half:]
    hr_data = hr[1:]
    hr_1, hr_2 = hr_data[:half], hr_data[half:]

    avg_hr_1 = np.mean(hr_1)
    avg_hr_2 = np.mean(hr_2)
    ef_1 = np.mean(pow_1) / avg_hr_1 if avg_hr_1 > 0 else 0.0
    ef_2 = np.mean(pow_2) / avg_hr_2 if avg_hr_2 > 0 else 0.0
    decoupling = ((ef_1 - ef_2) / ef_1 * 100) if ef_1 > 0 else 0.0

    return {
        "Znormalizowana moc (W)": round(float(normalized_power), 2),
        "Współczynnik wydajności (pierwsza połowa treningu)": round(float(ef_1), 3),
        "Współczynnik wydajności (druga połowa treningu)": round(float(ef_2), 3),
        "Aerobic decoupling": round(float(decoupling), 2)
    }
