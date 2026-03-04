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