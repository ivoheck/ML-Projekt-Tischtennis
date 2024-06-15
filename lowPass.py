import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Funktion zur Erzeugung eines Tiefpassfilters
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Funktion zur Anwendung des Tiefpassfilters auf die Daten
def apply_lowpass_filter(data, cutoff_frequency, sampling_frequency, filter_order=5):
    b, a = butter_lowpass(cutoff_frequency, sampling_frequency, order=filter_order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

data = pd.read_csv("daten/TischtennisTest1.csv", usecols=[2,3,4])
dataAccX = pd.read_csv("daten/TischtennisTest1.csv", usecols=[2])
#dataAccX = pd.read_csv("daten/2024-04-09_13_03_11_Apple Watch.csv", usecols=[13])

x = range(len(dataAccX))

cutoff_frequency = 100  # Grenzfrequenz des Filters in Hz
sampling_frequency = 500  # Abtastfrequenz der Daten in Hz
filter_order = 5  # Ordnung des Tiefpassfilters

gefilterte_daten = apply_lowpass_filter(dataAccX['accelerometerAccelerationX(G)'], cutoff_frequency, sampling_frequency, filter_order)

plt.plot(gefilterte_daten, label='Gefilterte Daten')

#Label anzeigen
plt.ylabel('accelerometerAccelerationX(G)')
ranges = [(224, 243), (284, 297), (324, 347), (366, 390), (406, 430), (453, 493)]

for start, end in ranges:
    plt.fill_betweenx([min(dataAccX['accelerometerAccelerationX(G)']), max(dataAccX['accelerometerAccelerationX(G)'])], start, end, color='red', alpha=0.3)



plt.plot(dataAccX, label='Originaldaten')
plt.show()
