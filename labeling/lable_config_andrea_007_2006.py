import pandas as pd
from enum import Enum, auto

from pandas import DataFrame


class Lable(Enum):
    vorhand = auto()
    rückhand = auto()
    schmetterball = auto()

    angabe_vorhand = auto()
    angabe_rückhand = auto()

    fail = auto()

time_colum = 'accelerometerTimestamp_sinceReboot(s)'

#Daten satz der gelabelt werden soll
data = pd.read_csv('C:/Users/m4ar/Downloads/OneDrive_2_27.6.2024/Watch_L_007_2006.csv')

clap_data = 370 #datenpunkt in dem das klatschen passiert
clap_video = [0,5,19] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,8,14,Lable.schmetterball],
        [0,10,7,Lable.schmetterball],
        [0,12,6,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [0,14,4,Lable.schmetterball],
        [0,16,2,Lable.schmetterball],
        [0,17,20,Lable.schmetterball],
        [0,19,6,Lable.schmetterball],
        [0,21,7,Lable.schmetterball],
        [0,23,3,Lable.schmetterball],
        [0,25,8,Lable.schmetterball],
        [0,27,2,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [0,28,19,Lable.schmetterball],
        [0,30,10,Lable.schmetterball],
        [0,32,4,Lable.schmetterball],
        [0,34,8,Lable.schmetterball],
        [0,36,6,Lable.schmetterball],
        [0,38,4,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [0,39,21,Lable.schmetterball],
        [0,41,16,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [0,43,9,Lable.schmetterball],
        [0,44,23,Lable.schmetterball],
        [0,46,16,Lable.schmetterball],
        [0,48,13,Lable.fail],
        [0,50,12,Lable.schmetterball],
        [0,52,10,Lable.schmetterball],
        [0,57,14,Lable.schmetterball],
        [0,59,5,Lable.schmetterball],
        [1,0,24,Lable.schmetterball],
        [1,3,2,Lable.schmetterball],
        [1,5,9,Lable.schmetterball],
        [1,7,1,Lable.schmetterball],
        [1,8,24,Lable.schmetterball],
        [1,10,16,Lable.schmetterball],
        [1,12,11,Lable.fail],
        [1,14,16,Lable.fail],
        [1,15,2,Lable.rückhand],
        [1,17,8,Lable.schmetterball],
        [1,20,4,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [1,21,24,Lable.schmetterball],
        [1,24,19,Lable.schmetterball],
        [1,26,15,Lable.schmetterball],
        [1,28,9,Lable.schmetterball],
        [1,30,10,Lable.schmetterball],
        [1,32,2,Lable.schmetterball],
        [1,33,16,Lable.schmetterball],
        [1,35,3,Lable.schmetterball],
        [1,36,19,Lable.schmetterball],
        [1,38,5,Lable.schmetterball],
        ] #Minute, Sekunde, Frame, Label