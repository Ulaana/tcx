import math
from datetime import datetime

R = 6371000


def distance(lat1, lon1, lat2, lon2):
    """
    Oblicza odległość między dwoma punktami GPS z wykorzystaniem wzoru Haversine'a.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def total_distance(latitudes, longitudes):
    """
    Oblicza całkowity pokonany dystans.
    Iteruje sekwencyjnie przez listę współrzędnych, sumując odległości
    między kolejnymi punktami pomiarowymi.
    """
    total_distance = 0.0
    for i in range(1, len(latitudes)):
        total_distance += distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
    return total_distance


def elevation_gain(elevations):
    """
    Oblicza całkowite przewyższenie (ang. elevation gain).
    Analizuje różnice wysokości krok po kroku i agreguje wyłącznie
    wartości dodatnie (reprezentujące ruch pod górę).
    """
    gain = 0.0
    for e1, e2 in zip(elevations[:-1], elevations[1:]):
        diff = e2 - e1
        if diff > 0:
            gain += diff
    return gain


def avg_hr(heart_rates):
    """
    Wylicza średnie tętno, filtrując szum z czujnika.
    Odrzuca wartości zerowe (brak sygnału z pulsometru) przed wykonaniem dzielenia,
    aby nie zaniżać rzeczywistego wyniku.
    """
    total_hr = 0
    count = 0
    for hr in heart_rates:
        if hr > 0:
            total_hr += hr
            count += 1
    return total_hr / count if count > 0 else 0


def hr_zones(heart_rates, hr_max=185):
    """
    Klasyfikuje tętno do 5 standardowych stref wysiłkowych.
    Progi stref wyliczane są jako procenty zdefiniowanego tętna maksymalnego
    (50%, 60%, 70%, 80%, 90%). Zwraca histogram w postaci listy.
    """
    z1_min, z2_min, z3_min, z4_min, z5_min = [hr_max * p for p in (0.5, 0.6, 0.7, 0.8, 0.9)]
    zones = [0, 0, 0, 0, 0]
    for hr in heart_rates:
        if hr >= z5_min:
            zones[4] += 1
        elif hr >= z4_min:
            zones[3] += 1
        elif hr >= z3_min:
            zones[2] += 1
        elif hr >= z2_min:
            zones[1] += 1
        elif hr >= z1_min:
            zones[0] += 1
    return zones


def elevation_hr(elevations, heart_rates):
    """
    Koreluje dane z pulsometru z profilem wysokościowym trasy.
    Rozdziela tętno na dwie kategorie: praca podczas wspinaczki (dodatnia różnica wzniesień)
    oraz odpoczynek/praca na zbiegach i płaskim (ujemna lub zerowa różnica).
    """
    climb_hr_total = 0
    climb_count = 0
    descent_hr_total = 0
    descent_count = 0
    for e1, e2, hr in zip(elevations[:-1], elevations[1:], heart_rates[1:]):
        ele_diff = e2 - e1
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


def perf_eff(latitudes, longitudes, elevations, heart_rates, times, weight=75.0):
    """
    Zaawansowana analiza wydolności biomechanicznej i tlenowej.
    1. Szacuje generowaną moc (power) na podstawie prędkości, masy ciała i nachylenia terenu.
    2. Oblicza moc znormalizowaną (algorytm średniej kroczącej 30s podniesionej do 4 potęgi).
    3. Wylicza 'Aerobic decoupling' – wskaźnik porównujący stosunek mocy do tętna z pierwszej
       połowy treningu do drugiej (pozwala ocenić narastające zmęczenie organizmu).
    """
    delta_times = []
    distances = []
    speeds = []
    grades = []
    powers = []

    for i in range(1, len(latitudes)):
        delta_time = times[i] - times[i - 1]
        if delta_time <= 0:
            delta_time = 1.0

        dist = distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
        ele_diff = elevations[i] - elevations[i - 1]
        speed = dist / delta_time
        grade = (ele_diff / dist) if dist > 0 else 0
        # Obliczanie estymowanej mocy chwilowej (w watach).
        # Wzór składa się z dwóch członów:
        # 1. Praca przeciwko grawitacji (masa * g * prędkość * nachylenie)
        # 2. Szacunkowy opór (0.5 * masa * prędkość)
        power = (weight * 9.81 * speed * grade) + (0.5 * weight * speed)
        power = max(0, power)

        delta_times.append(delta_time)
        distances.append(dist)
        speeds.append(speed)
        grades.append(grade)
        powers.append(power)

    power_4th = []
    # 30-sekundowe "okno" średniej kroczącej w celu wygładzenia
    window_size = 30
    for i in range(len(powers)):
        start = max(0, i - window_size + 1)
        window = powers[start:i + 1]
        rolling_mean = sum(window) / len(window)
        # Uśrednioną moc do 4. potęgi, aby nadać większą wagę
        # intensywnym, wyczerpującym momentom treningu.
        power_4th.append(rolling_mean ** 4)

    normalized_power = (sum(power_4th) / len(power_4th)) ** 0.25 if power_4th else 0
    half = len(powers) // 2

    # Funkcja pomocnicza: dzieli średnią moc przez średnie tętno,
    # określając, ile watów jest generowanych na jedno uderzenie serca.
    def calculate_ef(power, hr):
        avg_p = sum(power) / len(power) if power else 0
        avg_hr = sum(hr) / len(hr) if hr else 1
        return avg_p / avg_hr

    hr_data = heart_rates[1:]

    ef_1 = calculate_ef(powers[:half], hr_data[:half])
    ef_2 = calculate_ef(powers[half:], hr_data[half:])
    # Rozprzężenie (decoupling) to spadek wydajności w czasie wyrażony w procentach.
    decoupling = ((ef_1 - ef_2) / ef_1 * 100) if ef_2 > 0 else 0

    return {
        "Znormalizowana moc (W)": round(normalized_power, 2),
        "Współczynnik wydajności (pierwsza połowa treningu)": round(ef_1, 3),
        "Współczynnik wydajności (druga połowa treningu)": round(ef_2, 3),
        "Aerobic decoupling": round(decoupling, 2)
    }


def activity_segments(latitudes, longitudes, times, cadences, min_duration=5):
    """
    Segmentacja kinetyczna aktywności.
    Analizuje prędkość oraz kadencję (kroki), dzieląc nagranie na bloki:
    Postój (0), Chód (1), Bieg (2). Implementuje filtrację zakłóceń czasowych
    i agreguje połączone bloki w końcowy, czytelny raport.
    """
    n = len(times)
    dist = []
    for i in range(n - 1):
        distances = distance(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1])
        dist.append(distances)
    delta_time = []
    for i in range(n - 1):
        dt = times[i + 1] - times[i]
        delta_time.append(dt if dt != 0 else 1.0)
    speeds = [0.0]
    for i in range(n - 1):
        speeds.append(dist[i] / delta_time[i])

    states = []
    # Każdej sekundzie przypisywany jest określony stan aktywności
    # bazując na prędkości oraz kadencji (częstotliwości kroków).
    for i in range(n):
        speed = speeds[i]
        cadence = cadences[i]
        state = 0
        # Warunki dla chodu: prędkość od 0.5 m/s do 2.2 m/s (ok. 1.8-8 km/h)
        # lub jakakolwiek niewielka kadencja (poniżej 65 kroków/min na jedną nogę).
        if (0.5 < speed <= 2.2) or (0 < cadence < 65):
            state = 1
        # Warunki dla biegu: prędkość powyżej 2.2 m/s i kadencja powyżej 65 kroków/min na jedną nogę.
        if speed > 2.2 and cadence >= 65:
            state = 2
        # Jeśli urządzenie rejestruje prędkość, ale sensor nie odnotowuje żadnych kroków, jest to postój
        if 0.5 < speed < 3.0 and cadence == 0:
            state = 0
        states.append(state)

    # Szuka punktów granicznych (indeksów), w których zmienia się typ
    # aktywności (np. przejście z chodu do biegu).
    split_indices = [0]
    for i in range(1, n):
        if states[i] != states[i - 1]:
            split_indices.append(i)
    split_indices.append(n)

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
    # Jeśli wyodrębniony blok trwa krócej niż założone minimum
    # (np. 5 sekund fałszywego postoju w trakcie biegu),
    # przypisywany jest mu stan poprzedniego segmentu.
    filtered_segments = []
    for segment in segments:
        if segment['duration'] < min_duration and len(filtered_segments) > 0:
            segment['state'] = filtered_segments[-1]['state']
        filtered_segments.append(segment)
    # Scalanie sąsiadujących bloków mających ten sam stan w dłuższe i ciągłę bloki aktywności.
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

    # Przeliczanie danych na statystyki.
    for segment in final_segments:
        start_index = segment['start_idx']
        end_index = segment['end_idx'] - 1

        if start_index >= end_index:
            continue

        start_time_str = datetime.fromtimestamp(times[start_index]).strftime('%H:%M:%S')
        end_time_str = datetime.fromtimestamp(times[end_index]).strftime('%H:%M:%S')

        segment_dist = sum(dist[start_index:end_index])
        speed_slice = speeds[start_index:end_index + 1]
        avg_speed_kmh = (sum(speed_slice) / len(speed_slice)) * 3.6 if speed_slice else 0.0
        cadence_slice = cadences[start_index:end_index + 1]
        avg_cadence = sum(cadence_slice) / len(cadence_slice) if cadence_slice else 0.0

        results.append({
            'Typ': state_names[segment['state']],
            'Start': start_time_str,
            'Koniec': end_time_str,
            'Dystans (m)': round(segment_dist, 1),
            'Średnia prędkość (km/h)': round(avg_speed_kmh, 1),
            'Średnia kadencja (RPM)': int(round(avg_cadence, 0)) * 2
        })

    return results
