import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum, auto

import lable_config_linus_004 as lable_config

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

# Daten satz der gelabelt werden soll
data = lable_config.data

clap_data = lable_config.clap_data
clap_video = lable_config.clap_video
points = lable_config.points
video_fps = lable_config.video_fps

# Alles vor dem Klatschen wird gelöscht
data = data.iloc[clap_data:].copy()

datei_name = data['loggingTime(txt)'].iloc[0]
datei_name = datei_name.replace(':', '-')

clap_data_time = data[time_colum].iloc[clap_data]


def convert_to_seconds(data_point):
    minutes = float(data_point[0])
    seconds = float(data_point[1])
    frames = float(data_point[2])

    seconds += minutes * 60
    seconds += frames / video_fps
    return seconds


def check_for_timestamp_out_of_range(timestamp):
    max_time = data[time_colum].iloc[-1]
    return timestamp <= max_time


def label_at(data, seconds, label, marked_values):
    # Bei diesem Zeitstempel muss gelabelt werden
    timestamp = seconds + clap_data_time

    if check_for_timestamp_out_of_range(timestamp):
        index = (data[time_colum] - timestamp).abs().argmin()
        data.loc[index, 'label'] = label
        marked_values.append(index)
    else:
        print('Timestamp is out of range, cannot be labeled')


def label_data(data, points):
    data['label'] = None
    marked_values = []
    for point in points:
        seconds = convert_to_seconds(point[:3])
        seconds -= convert_to_seconds(clap_video)
        label = point[3].name

        if seconds > 0:
            label_at(data, seconds, label, marked_values)
    return marked_values


# Neue Kopie des DataFrame mit Labels
labeled_data = data.copy()
marked_values = label_data(labeled_data, points)

# Speichern der gelabelten Daten
labeled_data.to_csv(f'{datei_name}_linus_q.csv', index=True, header=True)
print(labeled_data['label'].dropna())

# Plotten der Daten mit markierten Werten
plt.plot(labeled_data['accelerometerAccelerationX(G)'])

for value in marked_values:
    plt.axvline(x=value, color='red', linestyle='--')

plt.xlabel('Index')
plt.show()