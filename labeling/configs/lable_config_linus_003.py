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
data = pd.read_csv(r"C:\Uni\\Watch_L_003_2006.csv", )

clap_data = 238 #datenpunkt in dem das klatschen passiert
clap_video = [0,4,9] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,8,21,Lable.rückhand],
        [0,10,13,Lable.rückhand],
        [0,12,7,Lable.rückhand],
        [0,13,24,Lable.rückhand],
        [0,15,14,Lable.rückhand],
        [0,19,3,Lable.rückhand],
        [0,20,20,Lable.rückhand],
        [0,22,14,Lable.rückhand],
        [0,24,8,Lable.rückhand],
        [0,29,15,Lable.angabe_rückhand],
        [0,31,11,Lable.fail],
        [0,36,1,Lable.angabe_rückhand],
        [0,37,12,Lable.rückhand],
        [0,38,22,Lable.rückhand],
        [0,40,14,Lable.rückhand],
        [0,42,10,Lable.rückhand],
        [0,37,23,Lable.rückhand],
        [0,43,23,Lable.fail],
        [0,46,2,Lable.angabe_rückhand],
        [0,46,17,Lable.rückhand],
        [0,50,10,Lable.rückhand],
        [0,52,0,Lable.rückhand],
        [0,54,0,Lable.vorhand],
        [0,55,14,Lable.rückhand],
        [0,57,6,Lable.rückhand],
        [0,59,6,Lable.rückhand],
        [1,0,21,Lable.rückhand],
        [1,2,15,Lable.rückhand],
        [1,4,4,Lable.rückhand],
        [1,12,20,Lable.rückhand],
        [1,14,13,Lable.rückhand],
        [1,16,5,Lable.rückhand],
        [1,17,17,Lable.rückhand],
        [1,19,7,Lable.rückhand],
        [1,23,1,Lable.rückhand],
        [1,25,1,Lable.vorhand],
        [1,31,9,Lable.rückhand],
        [1,32,22,Lable.rückhand],
        [1,37,21,Lable.rückhand],
        [1,39,10,Lable.rückhand],
        [1,40,22,Lable.rückhand],
        [1,42,12,Lable.rückhand],
        [1,44,1,Lable.rückhand],
        [1,45,19,Lable.rückhand],
        [1,47,11,Lable.rückhand],
        [1,49,3,Lable.rückhand],
        [1,50,16,Lable.rückhand],
        [1,52,1,Lable.rückhand],
        [1,53,13,Lable.rückhand],
        [1,57,5,Lable.rückhand],
        [1,59,1,Lable.rückhand],
        [2,3,18,Lable.rückhand],
        [2,5,10,Lable.rückhand],
        [2,7,0,Lable.rückhand],
        [2,8,15,Lable.rückhand],
        [2,10,10,Lable.fail],
        [2,15,2,Lable.rückhand],
        [2,16,15,Lable.rückhand],
        [2,18,7,Lable.rückhand],
        [2,19,22,Lable.rückhand],
        [2,21,11,Lable.rückhand],
        [2,22,24,Lable.rückhand],
        [2,24,17,Lable.rückhand],
        [2,26,15,Lable.rückhand],
        [2,28,13,Lable.rückhand],
        [2,30,2,Lable.fail],
        [2,34,14,Lable.angabe_rückhand],
        [2,36,4,Lable.rückhand],
        [2,37,22,Lable.rückhand],
        [2,39,14,Lable.rückhand],
        [2,41,3,Lable.rückhand],
        [2,42,20,Lable.rückhand],
        [2,44,8,Lable.rückhand],
        [2,45,24,Lable.rückhand],
        [2,50,5,Lable.rückhand],
        [2,52,1,Lable.rückhand],
        [2,53,16,Lable.fail],
        [2,57,5,Lable.angabe_rückhand],
        [3,3,10,Lable.rückhand],
        [3,4,20,Lable.rückhand],
        [3,6,5,Lable.rückhand],
        [3,7,18,Lable.rückhand],
        [3,9,12,Lable.rückhand],
        [3,10,23,Lable.rückhand],
        [3,12,10,Lable.rückhand],
        [3,13,23,Lable.rückhand],
        [3,15,9,Lable.rückhand],
        [3,16,22,Lable.rückhand],
        [3,18,9,Lable.rückhand],
        [3,19,20,Lable.rückhand],
        [3,21,8,Lable.rückhand],
        [3,22,24,Lable.rückhand],
        [3,24,12,Lable.fail],
        [3,28,9,Lable.angabe_rückhand],
]