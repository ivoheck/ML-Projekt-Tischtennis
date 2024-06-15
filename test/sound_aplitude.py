import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# MP3-Datei laden (angenommen, sie ist im selben Verzeichnis wie dieses Skript)
audio_file = AudioSegment.from_mp3("daten/test.mp3")

# Konvertiere die Audiodaten in ein numpy Array
data = np.array(audio_file.get_array_of_samples())

# Sampling-Rate der Audio-Datei
sampling_rate = audio_file.frame_rate

# Zeitvektor basierend auf der Länge der Audiodaten und der Sampling-Rate
time = np.arange(0, len(data)) / sampling_rate

# Plot erstellen
plt.figure(figsize=(10, 4))
plt.plot(time, data, color='b')

# Markiere Bereiche über einer Amplitude von 2000
threshold = 2000
above_threshold = data > threshold
plt.fill_between(time, data, threshold, where=above_threshold, color='r', alpha=0.3)

# Achsenbeschriftungen und Titel hinzufügen
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform with Amplitude > 2000 Marked')

# Anzeigen des Plots
plt.show()
