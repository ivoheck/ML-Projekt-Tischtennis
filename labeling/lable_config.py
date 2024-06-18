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

clap_data = None #datenpunkt in dem das klatschen passiert
clap_video = [] #Minuten (int), Sekunde(int) , frames(int) in dem das klatschen passiert

video_fps = 25

points = [
          ] #Minute, Sekunde, Frame, Label