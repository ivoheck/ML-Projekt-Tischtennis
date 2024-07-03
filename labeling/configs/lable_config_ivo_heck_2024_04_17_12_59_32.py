import pandas as pd
from enum import Enum, auto

#2024-04-17T12_59

class Lable(Enum):
    vorhand = auto()
    rückhand = auto()
    schmetterball = auto()

    angabe_vorhand = auto()
    angabe_rückhand = auto()

    fail = auto()

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = pd.read_csv("daten/TischtennisTest1.csv", )

clap_data = 162 #datenpunkt in dem das klatschen passiert
clap_video = [0,0,24] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,1,0,Lable.fail],
        [0,3,12,Lable.angabe_vorhand],
        [0,5,12,Lable.rückhand],
        [0,6,21,Lable.rückhand],
        [0,8,10,Lable.rückhand],
        [0,9,18,Lable.rückhand],
        [0,11,12,Lable.fail]
          ] #Minute, Sekunde, Frame, Label