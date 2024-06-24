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
data = pd.read_csv("daten/Watch_L_006_2006.csv", )

clap_data = 451 #datenpunkt in dem das klatschen passiert
clap_video = [0,3,19] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,3,20,Lable.fail],
        [0,7,19,Lable.angabe_rückhand],
        [0,9,8,Lable.rückhand],
        [0,11,3,Lable.vorhand],
        [0,12,15,Lable.rückhand],
        [0,14,8,Lable.rückhand],
        [0,15,22,Lable.rückhand],
        [0,17,12,Lable.rückhand],
        [0,19,6,Lable.vorhand],
        [0,21,2,Lable.rückhand],
        [0,22,15,Lable.rückhand],
        [0,28,10,Lable.angabe_vorhand],
        [0,32,13,Lable.rückhand],
        [0,34,1,Lable.rückhand],
        [0,35,20,Lable.vorhand],
        [0,37,13,Lable.vorhand],
        [0,46,22,Lable.rückhand],
        [0,48,20,Lable.rückhand],
        [0,50,9,Lable.rückhand],
        [0,51,22,Lable.rückhand],
        [0,53,24,Lable.angabe_rückhand],
        [0,58,8,Lable.vorhand],
        [0,59,23,Lable.rückhand],
        [1,1,14,Lable.vorhand],
        [1,3,3,Lable.rückhand],
        [1,4,19,Lable.vorhand],
        [1,6,9,Lable.rückhand],
        [1,7,21,Lable.rückhand],
        [1,9,14,Lable.vorhand],
        [1,11,1,Lable.rückhand],
        [1,12,21,Lable.vorhand],
        [1,20,4,Lable.rückhand],
        [1,21,20,Lable.rückhand],
        [1,23,11,Lable.rückhand],
        [1,25,6,Lable.vorhand],
        [1,26,19,Lable.rückhand],
        [1,28,11,Lable.rückhand],
        [1,29,24,Lable.rückhand],
        [1,31,21,Lable.rückhand],
        [1,39,8,Lable.rückhand],
        [1,42,9,Lable.vorhand],
        [1,46,21,Lable.rückhand],
        [1,48,9,Lable.fail],
        [1,53,13,Lable.angabe_rückhand],
        [1,55,1,Lable.rückhand],
        [1,56,13,Lable.rückhand],
        [2,2,3,Lable.vorhand],
          ] #Minute, Sekunde, Frame, Label