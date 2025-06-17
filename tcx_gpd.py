from lxml import etree
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

pd.set_option('display.max_columns', None)

#PARSER

def parse_tcx(file):
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
          'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'}
    tree = etree.parse(file)
    trackpoints = tree.findall('.//tcx:Trackpoint', ns)
    data = []
    for trackpoint in trackpoints:
        time_elem = trackpoint.find('tcx:Time', ns)
        hr_elem = trackpoint.find('tcx:HeartRateBpm/tcx:Value', ns)
        dist_elem = trackpoint.find('tcx:DistanceMeters', ns)
        ele_elem = trackpoint.find('tcx:AltitudeMeters', ns)
        lat_elem = trackpoint.find('tcx:Position/tcx:LatitudeDegrees', ns)
        lon_elem = trackpoint.find('tcx:Position/tcx:LongitudeDegrees', ns)
        run_cadence_elem = trackpoint.find('tcx:Extensions/ns3:TPX/ns3:RunCadence', ns)
        if time_elem is not None:
            time = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00"))
        else:
            time = None
        if hr_elem is not None and hr_elem.text is not None:
            hr = int(hr_elem.text)
        else:
            hr = None
        if dist_elem is not None and dist_elem.text is not None:
            dist = float(dist_elem.text)
        else:
            dist = None
        if ele_elem is not None and ele_elem.text is not None:
            elevation = float(ele_elem.text)
        else:
            elevation = None
        if lat_elem is not None and lat_elem.text is not None:
            lat = float(lat_elem.text)
        else:
            lat = None
        if lon_elem is not None and lon_elem.text is not None:
            lon = float(lon_elem.text)
        else:
            lon = None
        if run_cadence_elem is not None and run_cadence_elem.text is not None:
            run_cadence = int(run_cadence_elem.text)
        else:
            run_cadence = None
        if run_cadence is not None:
            total_cadence = run_cadence * 2
        else:
            total_cadence = None
        if lon is not None and lat is not None:
            geometry = Point(lon, lat)
        else:
            geometry = None
        if time and dist is not None:
            data.append({
                'time': time,
                'heart_rate': hr,
                'distance': dist,
                'elevation': elevation,
                "lat": lat,
                "lon": lon,
                'run_cadence': run_cadence,
                'total_cadence': total_cadence,
                'geometry': geometry
            })
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    gdf = gdf.dropna(subset=['time', 'distance', 'elevation'])
    gdf = gdf.sort_values(by='time').reset_index(drop=True)
    return gdf


#ANALYSES

def calculate_pace(df):
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['dist_diff'] = df['distance'].diff()
    pace_data = []
    for time, distance in zip(df['time_diff'], df['dist_diff']):
        if time and distance and distance > 0:
            pace = (time / 60) / (distance / 1000)
            if pace < 20:
                pace_data.append(pace)
            else:
                pace_data.append(None)
        else:
            pace_data.append(None)
    df['pace'] = pace_data
    return df

def smoothing(df, smoothing_window_size):
    pace_smoothed = []
    cadence_smoothed = []
    for i in range(len(df)):
        start = max(0, i - smoothing_window_size // 2)
        end = min(len(df), i + smoothing_window_size // 2 + 1)

        pace_sum = 0
        pace_count = 0
        for j in range(start, end):
            pace_value = df.iloc[j]['pace']
            if pace_value is not None and not np.isnan(pace_value):
                pace_sum += pace_value
                pace_count += 1
        if pace_count > 0:
            pace_avg = pace_sum / pace_count
        else:
            pace_avg = None
        pace_smoothed.append(pace_avg)

        cadence_sum = 0
        cadence_count = 0
        for j in range(start, end):
            cadence_value = df.iloc[j]['total_cadence']
            if cadence_value is not None and not np.isnan(cadence_value):
                cadence_sum += cadence_value
                cadence_count += 1
        if cadence_count > 0:
            cadence_avg = cadence_sum / cadence_count
        else:
            cadence_avg = None
        cadence_smoothed.append(cadence_avg)
    df['pace_smoothed'] = pace_smoothed
    df['cadence_smoothed'] = cadence_smoothed
    return df

def segmenting_activities(df, walk_pace_threshold, cadence_threshold):
    activity_type = []
    for i in range(len(df)):
        pace = df.iloc[i]['pace_smoothed']
        cadence = df.iloc[i]['cadence_smoothed']
        total_cadence = df.iloc[i]['total_cadence']
        if pace is not None and cadence is not None:
            if pace <= walk_pace_threshold and cadence >= cadence_threshold:
                activity_type.append('bieg')
            elif total_cadence == 0:
                activity_type.append('postój')
            else:
                activity_type.append('chód')
    df['activity_type'] = activity_type

    segment_ids = [0]
    segment_id = 0
    for i in range(1, len(df)):
        if df.iloc[i]['activity_type'] != df.iloc[i-1]['activity_type']:
            segment_id += 1
        segment_ids.append(segment_id)
    df['segment_id'] = segment_ids
    return df

def merging_segments(df, min_segment_duration):
    segment_data = []
    unique_segments = sorted(set(df['segment_id']))
    for segment_id in unique_segments:
        rows = df[df['segment_id'] == segment_id]
        start_time = rows['time'].iloc[0]
        end_time = rows['time'].iloc[-1]
        start_distance = rows['distance'].iloc[0]
        end_distance = rows['distance'].iloc[-1]
        duration = (end_time - start_time).total_seconds()
        distance = end_distance - start_distance
        activity_type = rows['activity_type'].iloc[0]
        segment_data.append({
            'segment_id': segment_id,
            'start_time': start_time,
            'end_time': end_time,
            'distance': distance,
            'duration': duration,
            'activity_type': activity_type
        })

    merged_segments = []
    i = 0
    while i < len(segment_data):
        current = segment_data[i]
        if current['duration'] >= min_segment_duration:
            merged_segments.append(current)
            i += 1
        else:
            if len(merged_segments) > 0:
                previous_segment = merged_segments[-1]
            else:
                previous_segment = None
            if i + 1 < len(segment_data):
                next_segment = segment_data[i + 1]
            else:
                next_segment = None
            if previous_segment and next_segment:
                if previous_segment['duration'] >= next_segment['duration']:
                    neighbor = previous_segment
                    merge = True
                else:
                    neighbor = next_segment
                    merge = False
            elif previous_segment:
                neighbor = previous_segment
                merge = True
            elif next_segment:
                neighbor = next_segment
                merge = False
            else:
                merged_segments.append(current)
                i += 1
                continue
            if merge:
                neighbor['end_time'] = current['end_time']
                neighbor['distance'] += current['distance']
                neighbor['duration'] += current['duration']
                i += 1
            else:
                current['end_time'] = next_segment['end_time']
                current['distance'] += next_segment['distance']
                current['duration'] += next_segment['duration']
                current['activity_type'] = next_segment['activity_type']
                merged_segments.append(current)
                i += 2

    final_segments = []
    for segment in merged_segments:
        if not final_segments:
            final_segments.append(segment)
        else:
            last_segment = final_segments[-1]
            segment['start_time'] = last_segment['end_time']
            segment['end_time'] = segment['start_time'] + pd.to_timedelta(segment['duration'], unit='s')
            final_segments.append(segment)
    return pd.DataFrame(final_segments)

def detect_activities(df, smoothing_window_size=5, walk_pace_threshold=9.0, cadence_threshold=140, min_segment_duration=20):
    df = calculate_pace(df)
    df = smoothing(df, smoothing_window_size)
    df = segmenting_activities(df, walk_pace_threshold, cadence_threshold)
    segments = merging_segments(df, min_segment_duration)
    return segments

def calculate_elevation_profile(df):

    df['elevation_diff'] = df['elevation'].diff().fillna(0)
    elevation_gain = []
    for diff in df['elevation_diff']:
        if diff > 0:
            elevation_gain.append(diff)
        else:
            elevation_gain.append(0)
    df['elevation_gain'] = elevation_gain
    elevation_loss = []
    for diff in df['elevation_diff']:
        if diff < 0:
            elevation_loss.append(-diff)
        else:
            elevation_loss.append(0)
    df['elevation_loss'] = elevation_loss
    return df

def pace_hr_correlation(df):
    result = {}

    duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
    halfway = df['time'].iloc[0] + pd.to_timedelta(duration / 2, unit='s')
    first_half = df[df['time'] <= halfway]
    second_half = df[df['time'] > halfway]

    result['mean_pace_fh'] = first_half['pace'].mean()
    result['mean_pace_sh'] = second_half['pace'].mean()
    result['mean_hr_fh'] = first_half['heart_rate'].mean()
    result['mean_hr_sh'] = second_half['heart_rate'].mean()

    if result['mean_pace_fh'] > result['mean_pace_sh']:
        result['split_type'] = 'negative split (przyspieszenie)'
    else:
        result['split_type'] = 'positive split (zwolnienie)'

    df_nonan = df.dropna(subset=['heart_rate'])
    if not df_nonan.empty:
        correlation = df_nonan['time'].astype(np.int64).corr(df_nonan['heart_rate'])
    else:
        correlation = None

    result['correlation'] = correlation
    return result

def compare_trainings(df1, df2):
    def compute_stats(df):
        duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        distance = df['distance'].iloc[-1] - df['distance'].iloc[0]
        avg_pace = df['pace'].mean()
        avg_hr = df['heart_rate'].mean()
        max_hr = df['heart_rate'].max()
        elevation_gain = df['elevation_gain'].sum()
        elevation_loss = df['elevation_loss'].sum()
        return {
            'Czas trwania treningu (min)': duration / 60,
            'Dystans (km)': distance / 1000,
            'Średnie tempo (min/km)': avg_pace,
            'Średnie tętno (bpm)': avg_hr,
            'Max. tętno (bpm)': max_hr,
            'Suma podbiegów [m]': elevation_gain,
            'Suma zbiegów [m]': elevation_loss
        }

    stats = {
        'Trening 1': compute_stats(df1),
        'Trening 2': compute_stats(df2)
    }

    comparison_df = pd.DataFrame(stats)
    comparison_df['Różnica'] = comparison_df['Trening 2'] - comparison_df['Trening 1']

    def evaluation(row):
        messages = []

        if row.name == 'Dystans (km)':
            if row['Różnica'] > 0:
                messages.append("Dłuższy dystans (+)")
            elif row['Różnica'] < 0:
                messages.append("Krótszy dystans (-)")

        elif row.name == 'Średnie tempo (min/km)':
            if row['Różnica'] < 0:
                messages.append("Lepsze tempo (+)")
            elif row['Różnica'] > 0:
                messages.append("Wolniejsze tempo (-)")

        elif row.name == 'Średnie tętno (bpm)':
            if row['Różnica'] < 0:
                messages.append("Niższe tętno (+)")
            elif row['Różnica'] > 0:
                messages.append("Wyższe tętno (-)")

        elif row.name == 'Max. tętno (bpm)':
            if row['Różnica'] < 0:
                messages.append("Niższe max tętno (+)")
            elif row['Różnica'] > 0:
                messages.append("Wyższe max tętno (-)")

        elif row.name == 'Czas trwania treningu (min)':
            if row['Różnica'] > 0:
                messages.append("Dłuższy trening")
            elif row['Różnica'] < 0:
                messages.append("Krótszy trening")

        return ', '.join(messages)

    comparison_df['Ocena'] = comparison_df.apply(evaluation, axis=1)

    print("Porównanie metryk treningowych:")
    print(comparison_df.round(2))

    return comparison_df


#PLOTS, RAPORTS


def plot_activities(segments):
    colors = {
        'bieg': '#0074D9',
        'chód': '#2ECC40',
        'postój': '#FF851B'
    }
    levels = {
        'bieg': 2,
        'chód': 4,
        'postój': 6
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in segments.iterrows():
        activity = row['activity_type']
        start = (row['start_time'] - segments.iloc[0]['start_time']).total_seconds() / 60
        duration = row['duration'] / 60
        level = levels[activity]
        color = colors.get(activity, '#AAAAAA')
        x = start + duration / 2
        ax.bar(
            x=x,
            height=level,
            width=duration,
            bottom=0,
            color=color,
            edgecolor='black',
            alpha=0.7
        )
    ax.set_yticks([2, 4, 6])
    ax.set_yticklabels(['Bieg', 'Chód', 'Postój'])
    ax.set_xlabel("Czas (minuty)")
    ax.set_xlim(0, (segments['end_time'].iloc[-1] - segments['start_time'].iloc[0]).total_seconds() / 60)
    ax.set_title("Wykres aktywności w trakcie treningu")
    plt.tight_layout()
    plt.show()

def plot_elevation_profile(df):
    df['distance_km'] = df['distance'] / 1000
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['distance_km'], df['elevation'], color='brown', linewidth=1.5)
    ax.set_xlabel("Dystans (km)")
    ax.set_ylabel("Wysokość n.p.m. (m)")
    ax.set_title("Profil wysokościowy trasy")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def pace_hr_corr_report(results):
    print(f"Analiza tempa i tętna podczas treningu:")
    print(f"- Średnia tempa w pierwszej połowie: {results['mean_pace_fh']:.2f} min/km")
    print(f"- Średnia tempa w drugiej połowie: {results['mean_pace_sh']:.2f} min/km")
    print(f"- Średnie tętno w pierwszej połowie: {results['mean_hr_fh']:.1f} bpm")
    print(f"- Średnie tętno w drugiej połowie: {results['mean_hr_sh']:.1f} bpm")
    print(f"- Typ splitu: {results['split_type']}")
    print( f"- Korelacja tempa z tętnem: {results['correlation']:.3f} (dodatnia korelacja - wzrost tętna/ujemna korelacja - spadek tętna)")


if __name__ == "__main__":
    tcx1 = parse_tcx("trening1.tcx")
    tcx2 = parse_tcx("trening2.tcx")
    tcx3 = parse_tcx("wypadek.tcx")
    calculate_pace(tcx1)
    calculate_pace(tcx2)
    calculate_elevation_profile(tcx1)
    calculate_elevation_profile(tcx2)
    compare_trainings(tcx1, tcx2)
    calculate_elevation_profile(tcx3)
    plot_elevation_profile(tcx3)
    result = pace_hr_correlation(tcx1)
    pace_hr_corr_report(result)
    df = detect_activities(tcx1)
    plot_activities(df)


