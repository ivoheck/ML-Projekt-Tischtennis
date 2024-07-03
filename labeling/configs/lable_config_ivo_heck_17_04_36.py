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
data = pd.read_csv("daten/stream Apple Watch 240617 17_04_36.csv", )

clap_data = 136 #datenpunkt in dem das klatschen passiert
clap_video = [0,4,20] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,4,21,Lable.fail],
        [0,8,2,Lable.angabe_vorhand],
          [0,9,23,Lable.fail],
          [0,16,1,Lable.angabe_vorhand],
          [0,17,21,Lable.vorhand],
          [0,19,10,Lable.rückhand],#fail
          [0,21,22, Lable.angabe_vorhand],
          [0,23,19, Lable.vorhand],#fail
          [0,25,13, Lable.angabe_vorhand],
          [0,27,13, Lable.vorhand],#fail
          [0,29,20, Lable.angabe_vorhand],
          [0,31,11,Lable.vorhand],
          [0,33,5,Lable.vorhand],
          [0,34,20,Lable.rückhand],
          [0,40,17,Lable.angabe_vorhand],
          [0,42,14,Lable.vorhand],
          [0,44,11,Lable.rückhand],
          [0,52,16,Lable.vorhand],
          [0,54,10,Lable.rückhand],
          [0,56,5,Lable.rückhand],#fail
          [1,8,17,Lable.angabe_vorhand],#fail
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
          [1,42,23,Lable.vorhand],
          [1,44,22,Lable.vorhand],#fail
          [1,47,8,Lable.angabe_vorhand],
          [1,49,3,Lable.vorhand],#fail
          [1,51,8,Lable.angabe_vorhand],
          [1,53,0,Lable.rückhand],
          [1,54,23,Lable.vorhand],
          [1,57,2,Lable.rückhand],
          [1,59,24,Lable.schmetterball],#fail
          [2,2,0,Lable.schmetterball],#fail
          [2,9,9,Lable.vorhand],
          [2,11,8,Lable.rückhand],
          [2,13,17,Lable.schmetterball],
          [2,23,20,Lable.vorhand],
          [2,25,9,Lable.rückhand],#fail
          [2,27,23,Lable.angabe_vorhand],
          [2,29,23,Lable.vorhand],
          [2,32,4,Lable.rückhand],
          [2,33,22,Lable.vorhand],
          [2,35,20,Lable.vorhand],#fail
          [2,38,9,Lable.angabe_vorhand],
          [2,40,10,Lable.vorhand],
          [2,41,23,Lable.rückhand],#fail
          [2,46,12,Lable.vorhand],
          [2,48,8,Lable.vorhand],#fail
          [2,50,10,Lable.angabe_vorhand],
          [2,52,12,Lable.rückhand],
          [3,4,18,Lable.vorhand],
          [3,6,10,Lable.rückhand],#fail
          [3,13,24,Lable.vorhand],
          [3,15,20,Lable.vorhand],
          [3,17,20,Lable.vorhand],
          [3,19,13,Lable.rückhand],
          [3,21,0,Lable.rückhand],
          [3,22,11,Lable.rückhand],#fail
          [3,24,10,Lable.angabe_vorhand],
          [3,26,9,Lable.rückhand],
          [3,28,5,Lable.vorhand],
          [3,36,0,Lable.vorhand],
          [3,37,9,Lable.rückhand],
          [3,43,10,Lable.vorhand],
          [3,44,23,Lable.rückhand],
          [3,46,10,Lable.rückhand],#fail
          [3,52,4,Lable.angabe_vorhand],
          [3,54,0,Lable.rückhand],
          [3,55,21,Lable.rückhand],
          [3,57,16,Lable.vorhand],
          [4,10,3,Lable.vorhand],
          [4,11,18,Lable.rückhand],
          [4,13,11,Lable.vorhand],
          [4,15,7,Lable.fail],#fail
          [4,16,20,Lable.angabe_rückhand],
          [4,18,12,Lable.rückhand],
          [4,19,24,Lable.vorhand],#fail
          [4,22,6,Lable.angabe_vorhand],
          [4,24,1,Lable.rückhand],
          [4,25,18,Lable.rückhand],
          [4,27,8,Lable.rückhand],
          [4,28,24,Lable.rückhand],
          [4,30,9,Lable.rückhand],#fail
          [4,32,4,Lable.angabe_vorhand],
          [4,34,1,Lable.rückhand],
          [4,35,11,Lable.rückhand],
          [4,37,1,Lable.rückhand],
          [4,38,10,Lable.rückhand],
          [4,40,5,Lable.fail],
          [4,43,21,Lable.angabe_vorhand],
          [4,45,20,Lable.vorhand],
          [4,47,19,Lable.vorhand],
          [4,49,21,Lable.vorhand],
          [4,51,18,Lable.rückhand],
          [4,53,10,Lable.rückhand],
          [4,55,2,Lable.rückhand],
          [5,3,8,Lable.vorhand],
          [5,4,21,Lable.rückhand],
          [5,6,3,Lable.fail],
          ] #Minute, Sekunde, Frame, Label