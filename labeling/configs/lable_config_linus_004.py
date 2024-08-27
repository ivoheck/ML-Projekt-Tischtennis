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

clap_data = 497 #datenpunkt in dem das klatschen passiert
clap_video = [0,5,10] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,8,5,Lable.angabe_rückhand],
        [0,10,0,Lable.vorhand],
        [0,12,6,Lable.fail],
        [0,13,24,Lable.angabe_rückhand],
        [0,16,7,Lable.rückhand],
        [0,18,1,Lable.rückhand],
        [0,19,22,Lable.fail],
        [0,26,24,Lable.rückhand],
        [0,24,8,Lable.rückhand],
        [0,28,19,Lable.rückhand],
        [0,30,18,Lable.rückhand],
        [0,33,0,Lable.vorhand],
        [0,41,17,Lable.rückhand],
        [0,44,7,Lable.schmetterball],
        [0,40,14,Lable.rückhand],
        [0,52,22,Lable.angabe_rückhand],
        [0,54,11,Lable.rückhand],
        [0,57,21,Lable.angabe_rückhand],
        [0,59,12,Lable.rückhand],
        [1,1,4,Lable.rückhand],
        [1,3,4,Lable.rückhand],
        [1,10,5,Lable.angabe_rückhand],
        [1,11,21,Lable.rückhand],
        [1,13,13,Lable.vorhand],
        [1,15,7,Lable.vorhand],
        [1,17,12,Lable.vorhand],
        [1,19,15,Lable.fail],
        [1,21,21,Lable.angabe_rückhand],
        [1,26,11,Lable.angabe_vorhand],
        [1,28,13,Lable.rückhand],
        [1,30,7,Lable.rückhand],
        [1,32,4,Lable.fail],
        [1,34,2,Lable.angabe_vorhand],
        [1,36,6,Lable.rückhand],
        [1,38,21,Lable.rückhand],
        [1,40,22,Lable.rückhand],
        [1,42,19,Lable.vorhand],
        [1,44,20,Lable.vorhand],
        [1,50,10,Lable.angabe_rückhand],
        [1,52,0,Lable.rückhand],
        [1,53,11,Lable.rückhand],
        [1,42,12,Lable.rückhand],
        [2,0,5,Lable.rückhand],
        [2,1,15,Lable.rückhand],
        [2,4,20,Lable.rückhand],
        [2,6,8,Lable.rückhand],
        [2,7,20,Lable.rückhand],
]