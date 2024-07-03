import pandas as pd
from enum import Enum, auto


class Lable(Enum):
    vorhand = auto()
    rückhand = auto()
    schmetterball = auto()

    angabe_vorhand = auto()
    angabe_rückhand = auto()

    fail = auto()

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = pd.read_csv("C:/Users/m4ar/Downloads/SensorLogFiles_my_iOS_device_240702_10-52-27/SensorLogFiles_my_iOS_device_240702_10-52-27/stream Apple Watch 240702 10_34_04.csv")

clap_data = 92 #datenpunkt in dem das klatschen passiert
clap_video = [0,5,16] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,13,16,Lable.schmetterball],
        [0,18,21,Lable.schmetterball],
        [0,21,1,Lable.schmetterball],
        [0,23,18,Lable.schmetterball],
        [0,26,3,Lable.schmetterball],
        [0,28,4,Lable.fail],
        [0,30,16,Lable.schmetterball],
        [0,32,11,Lable.fail],
        [0,34,8,Lable.schmetterball],
        [0,36,8,Lable.schmetterball],
        [0,38,8,Lable.schmetterball],
        [0,40,10,Lable.fail],
        [0,46,23,Lable.fail],
        [0,49,16,Lable.schmetterball],
        [0,51,13,Lable.schmetterball],
        [0,54,1,Lable.schmetterball],
        [0,55,21,Lable.schmetterball],
        [0,57,17,Lable.schmetterball],
        [0,59,8,Lable.schmetterball],
        [1,2,3,Lable.schmetterball],
        [1,6,4,Lable.schmetterball],
        [1,10,7,Lable.schmetterball],
        [1,11,24,Lable.schmetterball],
        [1,13,16,Lable.schmetterball],
        [1,15,8,Lable.schmetterball],
          ] #Minute, Sekunde, Frame, Label