import numpy as np

R = 6371000


def bbox_np(latitudes, longitudes):
    min_lat, min_lon = np.min(latitudes), np.min(longitudes)
    max_lat, max_lon = np.max(latitudes), np.max(longitudes)
    return (min_lat, min_lon, max_lat, max_lon)


def total_distance_np(lats, lons):
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    lat1, lon1 = lat_rad[:-1], lon_rad[:-1]
    lat2, lon2 = lat_rad[1:], lon_rad[1:]

    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1

    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * \
        np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = R * c
    return np.sum(distances)


def elevation_gain_np(elevations):
    diffs = np.diff(elevations)
    return np.sum(diffs[diffs > 0])


def avg_hr_np(heart_rates):
    valid_hr = heart_rates[heart_rates > 0]
    return np.mean(valid_hr) if valid_hr.size > 0 else 0


def hr_zones_np(heart_rates, hr_max=185):
    bins = [hr_max * 0.5, hr_max * 0.6, hr_max * 0.7, hr_max * 0.8, hr_max * 0.9, np.inf]
    counts, _ = np.histogram(heart_rates, bins=bins)
    return counts.tolist()


def elevation_hr_np(elevations, heart_rates):
    ele_diffs = np.diff(elevations)
    hr_segments = heart_rates[1:]
    valid_hr = hr_segments > 0
    is_climb = ele_diffs > 0
    is_descent_or_flat = ele_diffs <= 0
    hr_on_climbs = hr_segments[valid_hr & is_climb]
    hr_on_descents = hr_segments[valid_hr & is_descent_or_flat]
    avg_climb = np.mean(hr_on_climbs) if hr_on_climbs.size > 0 else 0
    avg_descent = np.mean(hr_on_descents) if hr_on_descents.size > 0 else 0
    return avg_climb, avg_descent


def perf_eff_np(lats, lons, elevations, heart_rates, times, weight=75.0):
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    delta_time = np.diff(times)
    delta_time = np.maximum(delta_time, 1.0)

    delta_lat = np.diff(lat_rad)
    delta_lon = np.diff(lon_rad)

    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(delta_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    dist = R * c
    ele_diff = np.diff(elevations)
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

    hr_data = heart_rates[1:]
    hr_1, hr_2 = hr_data[:half], hr_data[half:]

    avg_hr_1 = np.mean(hr_1) if hr_1.size > 0 else 0
    avg_hr_2 = np.mean(hr_2) if hr_2.size > 0 else 0

    ef_1 = np.mean(pow_1) / avg_hr_1 if avg_hr_1 > 0 else 0.0
    ef_2 = np.mean(pow_2) / avg_hr_2 if avg_hr_2 > 0 else 0.0
    decoupling = ((ef_1 - ef_2) / ef_1 * 100) if ef_1 > 0 else 0.0

    return {
        "Znormalizowana moc (W)": round(float(normalized_power), 2),
        "Współczynnik wydajności (pierwsza połowa treningu)": round(float(ef_1), 3),
        "Współczynnik wydajności (druga połowa treningu)": round(float(ef_2), 3),
        "Aerobic decoupling": round(float(decoupling), 2)
    }
