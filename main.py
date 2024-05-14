import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D


# Pfad zur CSV-Datei (ersetze diesen mit deinem tatsächlichen Dateipfad)
file_path = ('/Users/henrydoose/Desktop/MAchine Learning/TischtennisTest1.csv')

# Lade die Daten
data = pd.read_csv(file_path)

# Konvertiere den Zeitstempel in ein Datumsformat
data['loggingTime(txt)'] = pd.to_datetime(data['loggingTime(txt)'])

# Plot-Einstellungen für den Beschleunigungsmesser
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))

# Beschleunigungsmesser X-Achse
axes[0].plot(data['loggingTime(txt)'], data['accelerometerAccelerationX(G)'], label='Acceleration X (G)', color='blue')
axes[0].set_title('Accelerometer X-Axis over Time')
axes[0].set_ylabel('Acceleration (G)')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

# Beschleunigungsmesser Y-Achse
axes[1].plot(data['loggingTime(txt)'], data['accelerometerAccelerationY(G)'], label='Acceleration Y (G)', color='green')
axes[1].set_title('Accelerometer Y-Axis over Time')
axes[1].set_ylabel('Acceleration (G)')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

# Beschleunigungsmesser Z-Achse
axes[2].plot(data['loggingTime(txt)'], data['accelerometerAccelerationZ(G)'], label='Acceleration Z (G)', color='red')
axes[2].set_title('Accelerometer Z-Axis over Time')
axes[2].set_ylabel('Acceleration (G)')
axes[2].set_xlabel('Time')
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")

# Erstelle ein neues matplotlib Figure- und Axes-Objekt
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Daten für das 3D-Diagramm
x = data['accelerometerAccelerationX(G)']
y = data['accelerometerAccelerationY(G)']
z = data['accelerometerAccelerationZ(G)']

# Erstelle ein 3D-Streudiagramm
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Acceleration (G)')
ax.set_ylabel('Y Acceleration (G)')
ax.set_zlabel('Z Acceleration (G)')
ax.set_title('3D Visualization of Accelerometer Data')

plt.show()

plt.tight_layout()
plt.show()