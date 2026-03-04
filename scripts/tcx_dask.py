import dask.array as da
import numpy as np

R = 6371000

def bbox_da(points_arr):
    min_lat, min_lon = da.min(points_arr[:, :2], axis=0)
    max_lat, max_lon = da.max(points_arr[:, :2], axis=0)
    return min_lat, min_lon, max_lat, max_lon

def total_distance_da(points_arr):
    rad_arr = da.radians(points_arr[:, :2])
    lat1, lon1 = rad_arr[:-1, 0], rad_arr[:-1, 1]
    lat2, lon2 = rad_arr[1:, 0], rad_arr[1:, 1]

    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1

    a = da.sin(delta_phi / 2.0) ** 2 + \
        da.cos(lat1) * da.cos(lat2) * \
        da.sin(delta_lambda / 2.0) ** 2
    c = 2 * da.arctan2(da.sqrt(a), da.sqrt(1 - a))

    distances = R * c
    return da.sum(distances)

def elevation_gain_da(points_arr):
    diffs = da.diff(points_arr[:, 2])
    return da.sum(diffs[diffs > 0])

def avg_hr_da(points_arr):
    hr_arr = points_arr[:, 3]
    valid_hr = hr_arr[hr_arr > 0]
    return da.mean(valid_hr)

def hr_zones_da(points_arr, hr_max=185):
    hr_arr = points_arr[:, 3]
    bins = [hr_max * 0.5, hr_max * 0.6, hr_max * 0.7, hr_max * 0.8, hr_max * 0.9, np.inf]
    counts, _ = da.histogram(hr_arr, bins=bins)
    return counts

def elevation_hr_da(points_arr):
    ele_diffs = da.diff(points_arr[:, 2])
    hr_segments = points_arr[1:, 3]
    valid_hr = hr_segments > 0
    is_climb = ele_diffs > 0
    is_descent_or_flat = ele_diffs <= 0
    hr_on_climbs = hr_segments[valid_hr & is_climb]
    hr_on_descents = hr_segments[valid_hr & is_descent_or_flat]
    avg_climb = da.mean(hr_on_climbs) if hr_on_climbs.size > 0 else 0
    avg_descent = da.mean(hr_on_descents) if hr_on_descents.size > 0 else 0
    return avg_climb, avg_descent