import datetime
import math
import random


def generate(filename="data/plik_1000000.tcx", num_points=100_000):
    start_time = datetime.datetime.now(datetime.timezone.utc)
    center_lat = 52.0
    center_lon = 19.0
    radius = 0.5

    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">\n')
        f.write('  <Activities>\n')
        f.write('    <Activity Sport="Biking">\n')
        f.write(f'      <Id>{start_time.strftime("%Y-%m-%dT%H:%M:%SZ")}</Id>\n')
        f.write('      <Lap StartTime="{}">\n'.format(start_time.strftime("%Y-%m-%dT%H:%M:%SZ")))
        f.write('        <Track>\n')

        for i in range(num_points):
            current_time = start_time + datetime.timedelta(seconds=i)

            angle = (i % 100_000) / 100_000 * 2 * math.pi
            lat = center_lat + math.sin(angle) * radius
            lon = center_lon + math.cos(angle) * radius

            ele = 200 + math.sin(i / 5000) * 100 + random.uniform(-1, 1)

            hr = int(145 + math.sin(i / 2000) * 20 + random.randint(-2, 2))
            cadence = int(90 + random.randint(-5, 5))

            trackpoint = f"""          <Trackpoint>
            <Time>{current_time.strftime("%Y-%m-%dT%H:%M:%SZ")}</Time>
            <Position>
              <LatitudeDegrees>{lat:.6f}</LatitudeDegrees>
              <LongitudeDegrees>{lon:.6f}</LongitudeDegrees>
            </Position>
            <AltitudeMeters>{ele:.1f}</AltitudeMeters>
            <HeartRateBpm><Value>{hr}</Value></HeartRateBpm>
            <Cadence>{cadence}</Cadence>
          </Trackpoint>\n"""

            f.write(trackpoint)

            if (i + 1) % 100_000 == 0:
                print(f"Wygenerowano {i + 1} / {num_points} punktów...")

        f.write('        </Track>\n')
        f.write('      </Lap>\n')
        f.write('    </Activity>\n')
        f.write('  </Activities>\n')
        f.write('</TrainingCenterDatabase>\n')

    print(f"Zakończono! Plik {filename} jest gotowy do analizy.")


if __name__ == "__main__":
    generate(num_points=10_000_000)