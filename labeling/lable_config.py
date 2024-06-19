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
data = pd.read_csv("dateiname.csv", )

clap_data = [76] #datenpunkt in dem das klatschen passiert
clap_video = [0,13,6] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
        [0,20,12,Lable]
          ] #Minute, Sekunde, Frame, Label