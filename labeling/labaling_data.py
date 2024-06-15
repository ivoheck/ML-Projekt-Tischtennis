import pandas as pd
import matplotlib.pyplot as plt

labels = ['vorhand','rückhand']
time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = pd.read_csv("daten/TischtennisTest1.csv", usecols=[1,2,3,4])

clap_data = 162 #datenpunkt in dem das klatschen passiert
clap_video = [1,0] #Sekunde plus frames in dem das klatschen passiert
points = [[5,10,labels[0]],[6,23,labels[1]],[9,20,labels[1]]]
video_fps = 24.0

#Alles vor dem klatschen wird gelöscht
data = data.iloc[clap_data:]

clap_data_time = data[time_colum].iloc[clap_data]

def convert_to_seconds(data_point):
    seconds = float(data_point[0])
    frames = float(data_point[1])
    
    seconds += frames/video_fps
    return seconds

def label_at(seconds, label):
    #Bei diesem Zeitstempel muss gelabelt werden
    timestamp = seconds + clap_data_time

    index = (data[time_colum] - timestamp).abs().argmin()
    data.at[index,'label'] = label
    

def label_data(points):
    data['label'] = None
    for point in points:
        seconds = convert_to_seconds(point)
        seconds -= convert_to_seconds(clap_video)
        label = point[2]

        if seconds > 0:
            label_at(seconds, label)

label_data(points=points)
data.to_csv('labeld_dataframe.csv', index=True, header=True)
print(data['label'].dropna())
