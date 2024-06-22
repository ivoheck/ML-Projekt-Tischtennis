import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

def get_csv_from_directory(data_path):
    label_vorhand, label_rückhand, label_schmetterball, label_angabe_vorhand, label_angabe_rückhand = 0,0,0,0,0
    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        if os.path.isfile(path):  # Überprüfen, ob es eine Datei ist (kein Unterordner)

            df = pd.read_csv(path)
            label = df['label']

            label_vorhand += df['label'].str.contains('vorhand', case=False, na=False).sum()
            label_rückhand += df['label'].str.contains('rückhand', case=False, na=False).sum()
            label_schmetterball += df['label'].str.contains('schmetterball', case=False, na=False).sum()
            label_angabe_vorhand += df['label'].str.contains('angabe_vorhand', case=False, na=False).sum()
            label_angabe_rückhand += df['label'].str.contains('angabe_rückhand', case=False, na=False).sum()

    return label_vorhand, label_rückhand, label_schmetterball, label_angabe_vorhand, label_angabe_rückhand

data_path_30 = '../labeled_data_raw_30_herz/' #Ordner in dehm die roh daten liegen
data_path_90 = '../labeled_data_raw_90_herz/'
data_path_100 = '../labeled_data_raw_100_herz/'

label_vorhand, label_rückhand, label_schmetterball, label_angabe_vorhand, label_angabe_rückhand = 0,0,0,0,0

data_paths = [data_path_30, data_path_90, data_path_100]

for path in data_paths:
    c_vorhand, c_rückhand, c_schmetterball, c_angabe_vorhand, c_angabe_rückhand = get_csv_from_directory(path)
    label_vorhand += c_vorhand
    label_rückhand += c_rückhand
    label_schmetterball += c_schmetterball
    label_angabe_vorhand += c_angabe_vorhand
    label_angabe_rückhand += c_angabe_rückhand


labels = ['Vorhand', 'Rückhand', 'Schmetterball', 'Angabe Vorhand', 'Angabe Rückhand']
counts = [label_vorhand, label_rückhand, label_schmetterball, label_angabe_vorhand, label_angabe_rückhand]

# Balkendiagramm erstellen
plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=counts, palette='viridis')

# Diagramm anpassen
plt.title('Daten verteilung')
plt.xlabel('Schlagtyp')
plt.ylabel('Anzahl')
plt.show()