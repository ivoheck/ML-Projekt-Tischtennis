import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum, auto

label_schlag = ['vorhand','rückhand','schmetterball','schupfball']
label_schlag_fail = ['vorhand_fail','rückhand_fail','schmetterball_fail','schupfball_fail']

label_angabe = ['angabe_vorhand','angabe_rückhand']
label_angabe_fail = ['angabe_vorhand_fail','angabe_rückhand_fail']

class Lable(Enum):
    vorhand = auto()
    rückhand = auto()
    schmetterball = auto()
    schlupfball = auto()

    vorhand_fail = auto()
    rückhand_fail = auto()
    schmetterball_fail = auto()
    schlupfball_fail = auto()

    angabe_vorhand = auto()
    angabe_rückhand = auto()

    angabe_vorhand_fail = auto()
    angabe_rückhand_fail = auto()

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = pd.read_csv("stream Apple Watch 240617 17_04_36.csv", )

clap_data = 136 #datenpunkt in dem das klatschen passiert
clap_video = [4,20] #Sekunde(int) , frames(int) in dem das klatschen passiert
points = [[0,8,2,Lable.angabe_vorhand],
          [0,9,23,Lable.angabe_vorhand_fail],
          [0,16,1,Lable.angabe_vorhand],
          [0,17,21,Lable.schlupfball],
          [0,19,10,Lable.rückhand_fail],
          [0,21,22, Lable.angabe_vorhand],
          [0,23,19, Lable.vorhand_fail],
          [0,25,13, Lable.angabe_vorhand],
          [0,27,13, Lable.vorhand_fail],
          [0,29,20, Lable.angabe_vorhand],
          [0,31,11,Lable.vorhand],
          [0,33,5,Lable.vorhand],
          [0,34,20,Lable.rückhand],
          [0,40,17,Lable.angabe_vorhand],
          [0,42,14,Lable.vorhand],
          [0,44,11,Lable.rückhand],
          [0,52,16,Lable.vorhand],
          [0,54,10,Lable.rückhand],
          [0,56,5,Lable.rückhand_fail],
          [1,8,17,Lable.angabe_vorhand_fail],#hier
          [1,12,0,Lable.angabe_vorhand],
          [1,14,1,Lable.vorhand],
          [1,17,15,Lable.angabe_vorhand],
          [1,19,10,Lable.vorhand],
          [1,21,2,Lable.rückhand],
          [1,25,8,Lable.vorhand],
          [1,29,18,Lable.angabe_vorhand],
          [1,31,12,Lable.rückhand],
          [1,33,9,Lable.rückhand],
          [1,38,17,Lable.angabe_vorhand],
          [1,42,23,Lable.schlupfball],
          [1,44,22,Lable.schlupfball],
          [1,47,8,Lable.angabe_vorhand],
          [1,49,3,Lable.vorhand],#fail
          [1,51,8,Lable.angabe_vorhand],
          [1,53,0,Lable.rückhand],
          [1,54,23,Lable.vorhand],
          [1,57,2,Lable.rückhand],
          [1,59,24,Lable.schmetterball],#fail
          [2,2,0,Lable.schmetterball],#fail
          [2,9,9,Lable.schlupfball],#eindeutig
          [2,11,8,Lable.rückhand],
          [2,13,17,Lable.schmetterball],
          [2,23,20,Lable.schlupfball],
          [2,25,9,Lable.rückhand],#fail
          [2,27,23,Lable.angabe_vorhand],
          [2,29,23,Lable.schlupfball],
          ] #list<list<int,int,String>> sekunde, frame, label
video_fps = 25

#Alles vor dem klatschen wird gelöscht
data = data.iloc[clap_data:]

datei_name = data['loggingTime(txt)'].iloc[0]

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

    marked_values.append(index)
    

def label_data(points):
    data['label'] = None
    for point in points:
        seconds = convert_to_seconds(point)
        seconds -= convert_to_seconds(clap_video)
        label = point[2]

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
