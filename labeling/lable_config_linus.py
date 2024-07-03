import pandas as pd
from enum import Enum, auto

class Lable(Enum):
    vorhand = auto()
    rückhand = auto()
    schmetterball = auto()

    angabe_vorhand = auto()
    angabe_rückhand = auto()

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = pd.read_csv(r"C:\Uni\\Watch_L_001_2006.csv",)

clap_data = [315] #datenpunkt in dem das klatschen passiert
clap_video = [0,3,23] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,9,15,Lable.angabe_rückhand],
        [0,11,9,Lable.rückhand],
        [0,13,6,Lable.rückhand],
        [0,16,14,Lable.rückhand],
        [0,18,11,Lable.rückhand],
        [0,20,5,Lable.rückhand],
        [0,22,1,Lable.rückhand],
        [0,24,5,Lable.vorhand],
        [0,25,24,Lable.rückhand],
        [0,30,1,Lable.angabe_vorhand],
        [0,31,20,Lable.rückhand],
        [0,33,14,Lable.rückhand],
        [0,41,8,Lable.rückhand],
        [0,48,1,Lable.rückhand],
        [0,49,24,Lable.rückhand],
        [0,51,20,Lable.vorhand],
        [0,57,2,Lable.angabe_rückhand],
        [0,58,16,Lable.rückhand],
        [1,0,11,Lable.rückhand],
        [1,1,23,Lable.rückhand],
        [1,8,4,Lable.vorhand],
        [1,10,5,Lable.vorhand],
        [1,12,0,Lable.rückhand],
        [1,14,23,Lable.vorhand]
          ] #Minute, Sekunde, Frame, Label