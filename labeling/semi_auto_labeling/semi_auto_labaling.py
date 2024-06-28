import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tkinter as tk
from tkinter import simpledialog
from enum import Enum, auto

def eingabe_zahl_popup():
    # Initialisierung der Variable eingabe im äußeren Scope
    eingabe = None
    
    # Funktion zur Erstellung des Popup-Fensters
    def speichern_und_schliessen():
        nonlocal eingabe
        eingabe = entry.get()  # Den eingegebenen Text im Textfeld erhalten
        try:
            eingabe = int(eingabe)  # Die Eingabe in eine Ganzzahl konvertieren
            root.destroy()  # Das Hauptfenster schließen
        except ValueError:
            # Fehlerbehandlung, falls keine gültige Zahl eingegeben wurde
            label_status.config(text="Ungültige Eingabe. Bitte eine Zahl eingeben.")

    # Hauptfenster erstellen
    root = tk.Tk()
    root.title("Zahl eingeben")

    # Textfeld für die Eingabe erstellen
    entry = tk.Entry(root, width=20)
    entry.pack(pady=10)

    # Button zum Speichern und Schließen des Popup-Fensters
    button_speichern = tk.Button(root, text="Speichern", command=speichern_und_schliessen)
    button_speichern.pack(pady=10)

    # Statuslabel für Fehlermeldungen
    label_status = tk.Label(root, text="", fg="red")
    label_status.pack(pady=5)

    # Hauptfenster starten
    root.mainloop()

    # Nachdem das Hauptfenster geschlossen wurde, wird die eingegebene Zahl zurückgegeben
    return eingabe

def get_path(directory,file_ending):
    files =[]
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and path.endswith(file_ending):
            files.append(path)

    return files

def find_clap(directory):
    csv_files = get_path(directory=directory,file_ending='.csv')
            
    if len(csv_files) == 1:
        data = pd.read_csv(csv_files[0])

        #print(data['accelerometerAccelerationX(G)'])

        plt.plot(data['accelerometerAccelerationX(G)']) 

        plt.xlabel('Index') 
        plt.show()
    
def show_video_with_controls(video_path):
    # Video-Capture-Objekt erstellen
    cap = cv2.VideoCapture(video_path)

    # Überprüfen, ob das Video erfolgreich geöffnet wurde
    if not cap.isOpened():
        print(f"Fehler beim Öffnen des Videos von {video_path}")
        return

    # Schleife über alle Frames im Video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while current_frame < frame_count:
        # Frame an der aktuellen Position lesen
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        # Wenn das Lesen des Frames fehlschlägt, Schleife unterbrechen
        if not ret:
            break

        # Frame anzeigen
        cv2.imshow('Frame', frame)

        # Auf Tastatureingabe warten
        key = cv2.waitKey(0) & 0xFF

        # 'q' drücken, um die Schleife zu beenden
        if key == ord('q'):
            #TODO: evtl fragen ob wirklich abrechen
            break

        # 's' drücken, um die aktuelle Frame-Nummer zu speichern
        elif key == ord('1'):
            labels.append([current_frame,Lable.vorhand])
            print("\033[92m" + "Label: " + Lable.vorhand.name + " hinzugefügt bei frame: " + str(current_frame) + "\033[0m")

        elif key == ord('2'):
            labels.append([current_frame,Lable.rückhand])
            print("\033[92m" + "Label: " + Lable.rückhand.name + " hinzugefügt bei frame: " + str(current_frame) + "\033[0m")

        elif key == ord('3'):
            labels.append([current_frame,Lable.schmetterball])
            print("\033[92m" + "Label: " + Lable.schmetterball.name + " hinzugefügt bei frame: " + str(current_frame) + "\033[0m")

        elif key == ord('4'):
            labels.append([current_frame,Lable.angabe_vorhand])
            print("\033[92m" + "Label: " + Lable.angabe_vorhand.name + " hinzugefügt bei frame: " + str(current_frame) + "\033[0m")

        elif key == ord('5'):
            labels.append([current_frame,Lable.angabe_rückhand])
            print("\033[92m" + "Label: " + Lable.angabe_rückhand.name + " hinzugefügt bei frame: " + str(current_frame) + "\033[0m")

        elif key == ord('f'):
            labels.append([current_frame,Lable.fail])
            print("\033[92m" + "Label: " + Lable.fail.name + " hinzugefügt bei frame: " + str(current_frame) + "\033[0m")

        elif key == ord('c'):
            if len(clap_frame) == 0:
                clap_frame.append(current_frame)
                print("\033[92m" + "Klatschen hinzugefügt bei frame: " + str(current_frame) + "\033[0m")

            else:
                print("\033[93m" + "Klatschen wurde bereits gelabelt überschreiben mit: ü" + "\033[0m")

        elif key == ord('ü'):
            clap_frame.pop()
            clap_frame.append(current_frame)
            print("\033[92m" + "Klatschen überschrieben bei frame: " + str(current_frame) + "\033[0m")

        elif key == ord('d'):
            if len(labels) >= 1:
                labels.pop()
                print("\033[93m" + "Letztes Label wurde gelöscht" + "\033[0m")

            else:
                print("\033[93m" + "Keine Label Vorhande" + "\033[0m")

        elif key == 81:  # ASCII-Code für die linke Pfeiltaste
            current_frame -= 1
            print('test')

        elif key == 83: #rechte pfeiltaste
            current_frame += 1

        elif key == ord('+'):
            current_frame += 5
            print("\033[92m" + str(100 * (current_frame/frame_count)) + "%" + "\033[0m")

        elif key == ord('-'):
            current_frame -= 5

    # Video-Capture-Objekt und alle Fenster schließen
    cap.release()
    cv2.destroyAllWindows()

def check_for_timestamp_out_of_range(timestamp):
    max_time = data[time_colum].iloc[-1]
    return timestamp <= max_time

def frames_to_seconds(frames):
    return frames/fps

def label_at(seconds, label, clap_data_time):
    #Bei diesem Zeitstempel muss gelabelt werden
    timestamp = seconds + clap_data_time

    if check_for_timestamp_out_of_range(timestamp):

        index = (data[time_colum] - timestamp).abs().argmin()
        data.at[index,'label'] = label.name

        marked_values.append(index)

    else:
        print(timestamp)
        print('timestap is out of range cant be labled')

def generate_labeld_data(labels,clap_frame,clap_data_point):

    clap_data_time = data[time_colum].iloc[clap_data_point]

    for point in labels:
        seconds = frames_to_seconds(point[0])
        seconds -= frames_to_seconds(clap_frame[0])

        if seconds > 0:
            label_at(seconds, point[1],clap_data_time)


class Lable(Enum):
    vorhand = auto()
    rückhand = auto()
    schmetterball = auto()

    angabe_vorhand = auto()
    angabe_rückhand = auto()

    fail = auto()

directory = 'data/'
vidio_ending = '.MOV'
fps = 25
time_colum = 'accelerometerTimestamp_sinceReboot(s)'

labels = []
clap_frame = []
marked_values = []

csv_files = get_path(directory=directory,file_ending='.csv')
            
if len(csv_files) == 1:
    data = pd.read_csv(csv_files[0])
    data['label'] = None

def main():
    find_clap(directory=directory)
    clap_data_point = eingabe_zahl_popup()
    video_path = get_path(directory=directory,file_ending=vidio_ending)

    if len(video_path) == 1:
        show_video_with_controls(video_path=video_path[0])

    generate_labeld_data(labels,clap_frame,clap_data_point)


main()

datei_name = 'data001_2006'#data['loggingTime(txt)'].iloc[0]plt.axvline(x=x_position, color='red', linestyle='--', ymin=0.5)
data.to_csv(f'{datei_name}_ivo_heck.csv', index=True, header=True)
print(data['label'].dropna())

plt.plot(data['accelerometerAccelerationX(G)']) 

for value in marked_values:
    plt.axvline(x=value, color='red', linestyle='--') 

plt.xlabel('Index') 
plt.show()
