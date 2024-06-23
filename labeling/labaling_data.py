import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum, auto

import lable_config_ivo_heck_005_2006 as lable_config

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = lable_config.data

clap_data = lable_config.clap_data
clap_video = lable_config.clap_video
points = lable_config.points
video_fps = lable_config.video_fps

#Alles vor dem klatschen wird gelöscht
data = data.iloc[clap_data:]

datei_name = data['loggingTime(txt)'].iloc[0]

clap_data_time = data[time_colum].iloc[clap_data]

def convert_to_seconds(data_point):
    minutes = float(data_point[0])
    seconds = float(data_point[1])
    frames = float(data_point[2])
    
    seconds += minutes * 60
    seconds += frames/video_fps
    return seconds

def check_for_timestamp_out_of_range(timestamp):
    max_time = data[time_colum].iloc[-1]
    return timestamp <= max_time

def label_at(seconds, label):
    #Bei diesem Zeitstempel muss gelabelt werden
    timestamp = seconds + clap_data_time

    if check_for_timestamp_out_of_range(timestamp):

        index = (data[time_colum] - timestamp).abs().argmin()
        data.at[index,'label'] = label

        marked_values.append(index)

    else:
        print(timestamp)
        print('timestap is out of range cant be labled')
        
    

def label_data(points):
    data['label'] = None
    for point in points:
        seconds = convert_to_seconds(point)
        seconds -= convert_to_seconds(clap_video)
        label = point[3].name

        if seconds > 0:
            label_at(seconds, label)

marked_values = []
label_data(points=points)
data.to_csv(f'{datei_name}_ivo_heck.csv', index=True, header=True)
print(data['label'].dropna())

plt.plot(data['accelerometerAccelerationX(G)']) 

for value in marked_values:
    plt.axvline(x=value, color='red', linestyle='--') 

plt.xlabel('Index') 
plt.show()
