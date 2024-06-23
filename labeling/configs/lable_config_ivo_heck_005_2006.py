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
data = pd.read_csv("daten/Watch_L_005_2006.csv", )

clap_data = 727 #datenpunkt in dem das klatschen passiert
clap_video = [0,6,4] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
    [0,6,5,Lable.fail],
        [0,10,14,Lable.angabe_vorhand],
        [0,12,5,Lable.rückhand],
        [0,14,3,Lable.vorhand],
        [0,16,8,Lable.rückhand],
        [0,21,18,Lable.rückhand],
        [0,28,4,Lable.angabe_rückhand],
        [0,29,18,Lable.rückhand],
        [0,37,3,Lable.rückhand],
        [0,38,21,Lable.rückhand],
        [0,42,12,Lable.angabe_vorhand],
        [0,45,2,Lable.rückhand],
        [0,46,23,Lable.fail],
        [0,50,17,Lable.angabe_rückhand],
        [0,52,8,Lable.rückhand],
        [0,53,17,Lable.rückhand],
        [1,0,13,Lable.rückhand],
        [1,2,7,Lable.rückhand],
        [1,3,23,Lable.rückhand],
        [1,5,19,Lable.vorhand],
        [1,7,18,Lable.rückhand],
        [1,9,19,Lable.vorhand],
        [1,11,10,Lable.rückhand],
        [1,12,24,Lable.rückhand],
        [1,15,3,Lable.schmetterball],
        [1,22,12,Lable.rückhand],
        [1,14,10,Lable.rückhand],
        [1,26,4,Lable.rückhand],
        [1,27,8,Lable.fail],
        [1,30,7,Lable.angabe_vorhand],
        [1,31,23,Lable.rückhand],
        [1,33,24,Lable.rückhand],
        [1,35,23,Lable.rückhand],#ausserhalb sichtfeld
        [1,37,22,Lable.vorhand],
        [1,39,18,Lable.fail],
        [1,45,9,Lable.angabe_rückhand],
        [1,47,0,Lable.rückhand],
        [1,48,18,Lable.vorhand],
        [1,50,15,Lable.vorhand],
        [1,58,19,Lable.rückhand],
        [2,0,9,Lable.rückhand],
        [2,1,18,Lable.fail],
        [2,7,1,Lable.angabe_vorhand],
        [2,8,18,Lable.rückhand],
        [2,10,14,Lable.fail],
        [2,11,4,Lable.fail],
        [2,20,8,Lable.rückhand],
        [2,22,3,Lable.rückhand],
        [2,23,9,Lable.rückhand],
        [2,24,23,Lable.fail],
        [2,27,0,Lable.rückhand],
        [2,28,8,Lable.rückhand],
        [2,30,2,Lable.vorhand],
        [2,32,0,Lable.fail],
        [2,36,12,Lable.angabe_vorhand],
        [2,38,4,Lable.fail],
        [2,41,0,Lable.fail],
          ] #Minute, Sekunde, Frame, Label