from datetime import datetime
import numpy as np

R = 6371000

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


def distances_np(latitudes, longitudes):
    phi1, phi2 = np.radians(latitudes[:-1]), np.radians(latitudes[1:])
    delta_phi = np.radians(latitudes[1:] - latitudes[:-1])
    delta_lambda = np.radians(longitudes[1:] - longitudes[:-1])

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def activity_segments_np(latitudes, longitudes, times, cadences, min_duration=5):
    delta_time = np.diff(times)
    delta_time[delta_time == 0] = 1.0
    dist_array = distances_np(latitudes, longitudes)
    speeds = np.insert(dist_array / delta_time, 0, 0.0)

    states = np.zeros(len(times), dtype=int)
    walk_mask = ((speeds > 0.5) & (speeds <= 2.2)) | ((cadences > 0) & (cadences < 65))
    states[walk_mask] = 1
    run_mask = (speeds > 2.2) & (cadences >= 65)
    states[run_mask] = 2
    idle_mask = (speeds > 0.5) & (cadences == 0) & (speeds < 3.0)
    states[idle_mask] = 0

    changes = np.where(states[:-1] != states[1:])[0] + 1
    split_indices = np.concatenate(([0], changes, [len(states)]))

    segments = []
    for i in range(len(split_indices) - 1):
        start_index = split_indices[i]
        end_index = split_indices[i + 1]
        state = states[start_index]
        segments.append({
            'state': state,
            'start_idx': start_index,
            'end_idx': end_index,
            'duration': end_index - start_index
        })

    filtered_segments = []
    for segment in segments:
        if segment['duration'] < min_duration and len(filtered_segments) > 0:
            segment['state'] = filtered_segments[-1]['state']
        filtered_segments.append(segment)

    final_segments = []
    for segment in filtered_segments:
        if not final_segments:
            final_segments.append(segment)
        elif final_segments[-1]['state'] == segment['state']:
            final_segments[-1]['end_idx'] = segment['end_idx']
            final_segments[-1]['duration'] += segment['duration']
        else:
            final_segments.append(segment)

    state_names = {0: "Postój (Idle)", 1: "Chód (Walk)", 2: "Bieg (Run)"}
    results = []

    for segment in final_segments:
        start_index = segment['start_idx']
        end_index = segment['end_idx'] - 1
        start_time_str = datetime.fromtimestamp(times[start_index]).strftime('%H:%M:%S')
        end_time_str = datetime.fromtimestamp(times[end_index]).strftime('%H:%M:%S')

        if start_index >= end_index:
            continue

        segment_dist = np.sum(dist_array[start_index:end_index])
        avg_speed_kmh = (np.mean(speeds[start_index:end_index + 1])) * 3.6
        avg_cadence = np.mean(cadences[start_index:end_index + 1])

        results.append({
            'Typ': state_names[segment['state']],
            'Start': start_time_str,
            'Koniec': end_time_str,
            'Dystans (m)': round(segment_dist, 1),
            'Średnia prędkość (km/h)': round(avg_speed_kmh, 1),
            'Średnia kadencja (RPM)': int(round(avg_cadence, 0))*2
        })

    return results
