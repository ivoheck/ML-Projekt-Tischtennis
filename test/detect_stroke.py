import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Daten einlesen
data = pd.read_csv("daten/TischtennisTest1.csv", usecols=[2,3,4])
dataAccX = pd.read_csv("daten/TischtennisTest1.csv", usecols=[2])

# x-Achse Werte
x = range(len(dataAccX))

# Butterworth Filter für Glättung der Daten (optional)
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Filter Parameter
cutoff = 2.0  # Cutoff frequency
fs = 50.0  # Sampling frequency
order = 2  # Filter order

# Glättung der Daten (optional)
dataAccX['smoothed'] = butter_lowpass_filter(dataAccX['accelerometerAccelerationX(G)'], cutoff, fs, order)

# Steigung berechnen
dataAccX['slope'] = np.gradient(dataAccX['smoothed'])

# Bedingung für die Steigung festlegen
threshold = 0.02
dataAccX['marker'] = dataAccX['slope'].apply(lambda x: 'Marker' if x > threshold else '')

# Plotting
plt.figure(figsize=(14, 7))

# Originaldaten plotten
plt.plot(dataAccX['accelerometerAccelerationX(G)'], label='Originaldaten')

# Markierte Bereiche hervorheben
ranges = [(224, 243), (284, 297), (324, 347), (366, 390), (406, 430), (453, 493)]
for start, end in ranges:
    plt.fill_betweenx([min(dataAccX['accelerometerAccelerationX(G)']), max(dataAccX['accelerometerAccelerationX(G)'])], start, end, color='red', alpha=0.3)

# Punkte mit hoher Steigung markieren
high_slope_points = dataAccX[dataAccX['marker'] == 'Marker']
plt.scatter(high_slope_points.index, high_slope_points['accelerometerAccelerationX(G)'], color='blue', label='Hohe Steigung', zorder=5)

# Beschriftungen und Legende
plt.xlabel('Zeit (samples)')
plt.ylabel('accelerometerAccelerationX(G)')
plt.title('Beschleunigungsdaten mit markierten Bereichen und hohen Steigungen')
plt.legend()
plt.show()
