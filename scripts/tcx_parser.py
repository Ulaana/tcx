from lxml import etree
from datetime import datetime
import numpy as np


def parser_py(file):
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
    latitudes, longitudes, elevations, heart_rates, times, cadences = [], [], [], [], [], []
    tree = etree.parse(file)

    for tp in tree.iterfind('.//tcx:Trackpoint', namespaces=ns):
        pos = tp.find('tcx:Position', namespaces=ns)
        if pos is not None:
            lat_elem = pos.find('tcx:LatitudeDegrees', namespaces=ns)
            lon_elem = pos.find('tcx:LongitudeDegrees', namespaces=ns)
            if lat_elem is not None and lon_elem is not None:
                time_elem = tp.find('tcx:Time', namespaces=ns)
                if time_elem is not None:
                    time_str = time_elem.text.replace('Z', '+00:00')
                    times.append(datetime.fromisoformat(time_str).timestamp())
                else:
                    times.append(0.0)
                latitudes.append(float(lat_elem.text))
                longitudes.append(float(lon_elem.text))
                ele_elem = tp.find('tcx:AltitudeMeters', namespaces=ns)
                elevations.append(float(ele_elem.text) if ele_elem is not None else 0.0)
                hr_elem = tp.find('.//tcx:HeartRateBpm/tcx:Value', namespaces=ns)
                heart_rates.append(float(hr_elem.text) if hr_elem is not None else 0.0)
                cad_elem = tp.find('tcx:Cadence', namespaces=ns)
                cadences.append(float(cad_elem.text) if cad_elem is not None else 0.0)

    return latitudes, longitudes, elevations, heart_rates, times, cadences


def parser_np(file):
    ns = "{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}"
    tree = etree.parse(file)
    trackpoints = tree.findall(f'.//{ns}Trackpoint')
    n = len(trackpoints)
    latitudes = np.empty(n, dtype=np.float64)
    longitudes = np.empty(n, dtype=np.float64)
    elevations = np.empty(n, dtype=np.float64)
    heart_rates = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)
    cadences = np.empty(n, dtype=np.float64)

    i = 0
    for tp in trackpoints:
        pos = tp.find(f'{ns}Position')
        if pos is not None:
            lat_str = pos.findtext(f'{ns}LatitudeDegrees')
            lon_str = pos.findtext(f'{ns}LongitudeDegrees')
            if lat_str is not None and lon_str is not None:
                time_str = tp.findtext(f'{ns}Time')
                if time_str is not None:
                    times[i] = datetime.fromisoformat(time_str.replace('Z', '+00:00')).timestamp()
                else:
                    times[i] = np.nan
                latitudes[i] = float(lat_str)
                longitudes[i] = float(lon_str)
                ele_str = tp.findtext(f'{ns}AltitudeMeters')
                elevations[i] = float(ele_str) if ele_str is not None else np.nan
                hr_str = tp.findtext(f'.//{ns}HeartRateBpm/{ns}Value')
                heart_rates[i] = float(hr_str) if hr_str is not None else np.nan
                cad_str = tp.findtext(f'{ns}Cadence')
                cadences[i] = float(cad_str) if cad_str is not None else np.nan
                i += 1

    return latitudes[:i], longitudes[:i], elevations[:i], heart_rates[:i], times[:i], cadences[:i]
