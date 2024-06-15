from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

# MP3-Datei einlesen
audio = AudioSegment.from_mp3("daten/test.mp3")

# Audiodaten extrahieren
samples = np.array(audio.get_array_of_samples())

# Falls die Audiodatei stereo ist, konvertieren wir sie zu mono
if audio.channels == 2:
    samples = samples.reshape((-1, 2))
    samples = samples.mean(axis=1)

# Zeitachse erstellen
sample_rate = audio.frame_rate
times = np.arange(len(samples)) / float(sample_rate)

# Fenstergröße und Schrittweite definieren (in Samples)
window_size = int(sample_rate * 0.5)  # 0.5 Sekunden Fenster
step_size = int(sample_rate * 0.05)   # 0.05 Sekunden Schritt

# Liste zum Speichern der Indizes der Fenster, die 75 Hz enthalten
marked_indices = []

# Minimaler Amplitudenwert für die 75 Hz Frequenz
min_amplitude = 5000  # Beispielwert, anpassen je nach Bedarf

# Frequenzanalyse über kleine Zeitfenster
for start in range(0, len(samples) - window_size, step_size):
    end = start + window_size
    window_samples = samples[start:end]
    
    # Fourier-Transformation
    frequencies = np.fft.fftfreq(len(window_samples), 1/sample_rate)
    fft_magnitude = np.abs(np.fft.fft(window_samples))
    
    # Nur die positiven Frequenzen betrachten
    positive_freqs = frequencies[:len(frequencies)//2]
    positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
    
    # Prüfen, ob 75 Hz in diesem Fenster vorhanden sind und die Amplitude größer als der Schwellenwert ist
    # Finde die Indexe der Frequenzen, die nahe bei 75 Hz sind
    freq_75_indices = np.where(np.isclose(positive_freqs, 75, atol=2.5))[0]

    # Prüfe, ob mindestens eine der gefundenen Frequenzen die Amplitudenbedingung erfüllt
    if any(positive_magnitude[freq_75_index] > min_amplitude for freq_75_index in freq_75_indices):
        marked_indices.append((start, end))

# Plotting der Originaldaten
plt.figure(figsize=(14, 7))
plt.plot(times, samples, label='Originaldaten')

# Markierte Bereiche hervorheben
for start, end in marked_indices:
    plt.axvspan(start / sample_rate, end / sample_rate, color='red', alpha=0.3)




# Beschriftungen und Legende
plt.xlabel('Zeit (s)')
plt.ylabel('Amplitude')
plt.title('Audiodaten mit markierten Bereichen (75 Hz)')
plt.legend()
plt.show()
