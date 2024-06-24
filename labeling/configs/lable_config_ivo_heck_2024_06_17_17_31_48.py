import pandas as pd
from enum import Enum, auto

#17_31

class Lable(Enum):
    vorhand = auto()
    rückhand = auto()
    schmetterball = auto()

    angabe_vorhand = auto()
    angabe_rückhand = auto()

    fail = auto()

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = pd.read_csv("daten/stream Apple Watch 240617 17_31_45.csv", )

clap_data = 76 #datenpunkt in dem das klatschen passiert
clap_video = [0,4,7] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,7,10,Lable.schmetterball],
        [0,10,13,Lable.schmetterball],
        [0,12,23,Lable.schmetterball],
        [0,15,18,Lable.schmetterball],
        [0,18,17,Lable.schmetterball],
        [0,21,1,Lable.schmetterball],
        [0,23,12,Lable.schmetterball],
        [0,26,14,Lable.schmetterball],
        [0,29,21,Lable.schmetterball],
        [0,32,23,Lable.schmetterball],
        [0,35,21,Lable.schmetterball],
        [0,38,24,Lable.schmetterball]
          ] #Minute, Sekunde, Frame, Label