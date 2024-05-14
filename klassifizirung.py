import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("daten/TischtennisTest1.csv", usecols=[2,3,4])

#Label spalte hinzufügen
data['label'] = 'unbekant'

#Angabe: 224 - 243
#Vorderhand: 284 - 297
#Rückhand: 324 - 347, 366 - 390, 406 - 430
#Schlag ins Leere: 453 - 493

data.loc[224:243, 'label'] = 'angabe'
data.loc[284:297, 'label'] = 'vorhand'
data.loc[324:347, 'label'] = 'rückhand'
data.loc[366:390, 'label'] = 'rückhand'
data.loc[406:430, 'label'] = 'rückhand'
data.loc[453:493, 'label'] = 'lehrschlag'

X = data[['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)','accelerometerAccelerationZ(G)']]  # Features (Beschleunigungsdaten)
y = data['label']  # Labels

rf_model = RandomForestClassifier()
rf_model.fit(X, y)

data_new = pd.read_csv("daten/2024-04-09_12_44_46_my_iOS_device.csv", usecols=[21,22,23])
print(data_new)
X_new = data_new[['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)','accelerometerAccelerationZ(G)']] 
#data['label'] = 'unbekannt'

predictions = rf_model.predict(X_new)
print(predictions)
#accuracy = accuracy_score(y, predictions)
#print("Genauigkeit des Modells auf den Trainingsdaten:", accuracy)

#klatschen 1 (162)
#Sekunde plus frames
#Angabe 03:02 - 3:18
#Vorderhand 5:04 - 5:16
#Rückhand 6:14 - 7:09
#Rückhand 8:01 - 08:22
#Rückhand 9:11 - 10:07
#Schlag ins lehre 11:02 - 12:12

#28 herz
#Angabe 02:02 - 2:18
#Vorderhand 4:04 - 4:16
#Rückhand 5:14 - 6:09
#Rückhand 7:01 - 07:22
#Rückhand 8:11 - 9:07
#Schlag ins lehre 10:02 - 11:12

#Angabe 2:08 - 2:72
#Vorderhand 4:16 - 4:64
#Rückhand 5:56 - 6:36
#Rückhand 7:04 - 07:88
#Rückhand 8:44 - 9:28
#Schlag ins lehre 10:08 - 11:48

#26.589 (klatschen [s])
#Angabe 28.669 - 29.309
#Vorderhand 30.749 - 31.229
#Rückhand 32.149 - 32.949
#Rückhand 33.629 - 34.469
#Rückhand 35.029 - 35.869
#Schlag ins lehre 36.669 - 38.069

#Angabe 62 - 81
#Vorderhand 122 - 135
#Rückhand 162 - 185
#Rückhand 204 - 228
#Rückhand 244 - 268
#Schlag ins lehre 291 - 331