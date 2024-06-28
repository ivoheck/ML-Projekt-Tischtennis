from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np
import os
import pandas as pd

def get_data(data_path):
    csv_file = []
    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        if os.path.isfile(path):
            return pd.read_csv(path)
        
def check_consecutive_predictions(predictions, index, step_size):
    return predictions[index] == predictions[index + step_size] == predictions[index + 2 * step_size]

# Funktion zum Gruppieren von Vorhersagen
def group_predictions(predictions, distance_threshold):
    grouped_predictions = []
    current_group = []
    
    for i in range(len(predictions)):
        if len(current_group) == 0:
            current_group.append((i, predictions[i]))
        else:
            last_index, last_prediction = current_group[-1]
            if i - last_index <= distance_threshold and predictions[i] == last_prediction:
                current_group.append((i, predictions[i]))
            else:
                if len(current_group) > 0:
                    # Füge die aktuelle Gruppe hinzu und starte eine neue Gruppe
                    grouped_predictions.append(current_group)
                current_group = [(i, predictions[i])]

    # Füge die letzte Gruppe hinzu
    if len(current_group) > 0:
        grouped_predictions.append(current_group)
    
    return grouped_predictions
        
def plot_data(df,predictions_df):
    plt.plot(df['accelerometerAccelerationX(G)'])
    
    predictions = predictions_df['Prediction'].tolist()
    
    grouped_predictions = group_predictions(predictions, distance_threshold)
    
    for group in grouped_predictions:
        if len(group) >= batch_size:  # Nur Gruppen mit mindestens 3 Vorhersagen plotten
            first_index, prediction = group[0]
            x_position = first_index * step_size + (num_rows / 2)
            if prediction == 0:  # vorhand
                plt.axvline(x=x_position, color='red', linestyle='--', ymax=0.5)
            elif prediction == 1:  # rückhand
                plt.axvline(x=x_position, color='green', linestyle='--' ,ymax=0.5)
            elif prediction == 2:  # schmetterball
                plt.axvline(x=x_position, color='blue', linestyle='--',ymax=0.5)
            #elif prediction == 3:
            #    plt.axvline(x=x_position, color='yellow', linestyle='--')

    true_lable = df['label'].tolist()

    for i,lable in enumerate(true_lable):
        print(i)
        if lable == 'Lable.rückhand':
           print(i)
           plt.axvline(x=i, color='red', linestyle='--', ymin=0.5)

    
    plt.xlabel('Index')
    plt.show()

data_path = 'test_data'
num_rows = 64
step_size = 6
batch_size = 5
distance_threshold= batch_size * step_size
feature_list = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)']

extracts = []

def main():
    df = get_data(data_path)

    for start in range(0, len(df) - num_rows + 1, step_size):
        extract = df.iloc[start:start + num_rows][feature_list].values
        extracts.append(extract)

    X_test = np.array(extracts)

    print(X_test.shape) 

    # Laden des gesamten Modells
    loaded_model = keras.models.load_model('model.h5')

    # Das Modell kann jetzt verwendet werden, um Vorhersagen zu treffen oder weiter trainiert zu werden
    loaded_model.summary()

    y_predicted = np.argmax(loaded_model.predict(x=X_test), axis=1)

    predictions_df = pd.DataFrame(y_predicted, columns=['Prediction'])

    plot_data(df,predictions_df)

main()