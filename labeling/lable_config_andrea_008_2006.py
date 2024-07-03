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
data = pd.read_csv("C:/Users/m4ar/Downloads/OneDrive_2_27.6.2024/Watch_L_008_2006.csv")

clap_data = 544  #datenpunkt in dem das klatschen passiert
clap_video = [0,4,7] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,15,22,Lable.schmetterball],
        [0,17,20,Lable.schmetterball],
        [0,19,20,Lable.schmetterball],
        [0,22,2,Lable.schmetterball],
        [0,23,12,Lable.schmetterball],
        [0,25,13,Lable.schmetterball],
        [0,27,10,Lable.schmetterball],
        [0,29,6,Lable.schmetterball],
        [0,31,2,Lable.schmetterball],
        [0,33,0,Lable.schmetterball],
        [0,34,24,Lable.schmetterball],
        [0,36,19,Lable.schmetterball],
        [0,54,18,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [0,56,10,Lable.fail],
        [0,58,8,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [0,59,23,Lable.schmetterball],
        [1,1,18,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [1,3,11,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [1,5,23,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [1,7,17,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [1,9,17,Lable.schmetterball],
        [1,11,12,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
        [1,13,9,Lable.schmetterball],
        [1,15,15,Lable.schmetterball],
        [1,17,16,Lable.schmetterball],#wahrscheinlich war außerhalb der Kamera
          ] #Minute, Sekunde, Frame, Label