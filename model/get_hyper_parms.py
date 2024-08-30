from matplotlib import units
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import random
import torch
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
keras.utils.set_random_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def extract_data(label,last_index,data,numpy_set,herz):
    for index in label:
        if herz == 30:
            if index - hit_duration/2 >= 0 and index + hit_duration/2 <= last_index:
                data_set = data[index-int(hit_duration/2):index+int(hit_duration/2)+1]
                numpy_data_set = data_set.values
                numpy_set = np.append(numpy_set, [numpy_data_set], axis=0).astype(np.float32) 

        elif herz == 90:
            if index - (hit_duration/2)*3 >= 0 and index + (hit_duration/2)*3 <= last_index:
                data_set = data[index-int(hit_duration/2)*3:index+int(hit_duration/2)*3+3]

                data_set = data_set.groupby(np.arange(len(data_set)) // 3).mean()

                numpy_data_set = data_set.values
                numpy_set = np.append(numpy_set, [numpy_data_set], axis=0).astype(np.float32) 
        
        #TODO: find better way for doing this
        elif herz == 100:
            if index - (hit_duration/2)*3 >= 0 and index + (hit_duration/2)*3 <= last_index:
                data_set = data[index-int(hit_duration/2)*3:index+int(hit_duration/2)*3+3]

                data_set = data_set.groupby(np.arange(len(data_set)) // 3).mean()

                numpy_data_set = data_set.values
                numpy_set = np.append(numpy_set, [numpy_data_set], axis=0).astype(np.float32) 

        else:
            print(herz, 'herz is not supported')

    return numpy_set

def extract_data_no_hit(label,last_index,data,numpy_set,max_label,herz):
    #index is the index of the label in the data set, i is the index of the list
    count = 0
    for i,index in enumerate(label):
        #TODO: handel first case
        if i == 0:
            pass

        elif i + 1 <= len(label) -1:
            prev = index

            while label[i+1] - prev >= 4 * hit_duration + 1:
                count += 1
                prev += 2 * int(hit_duration) + 1

                data_set = data[prev + int(hit_duration/2) : prev+int(hit_duration) + int(hit_duration/2) + 1]
                numpy_data_set = data_set.values
                numpy_set = np.append(numpy_set, [numpy_data_set], axis=0).astype(np.float32)

                #if count >= max_label*1:
                #   return numpy_set

    return numpy_set

def get_csv_from_directory(data_path,herz,numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball, numpy_set_kein_schlag):
    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        if os.path.isfile(path):  # Überprüfen, ob es eine Datei ist (kein Unterordner)

            df = pd.read_csv(path)
            data = df[feature_list]
            label = df['label']

            last_index = label.shape[0] - 1

            label_vorhand = df[df['label'].str.contains('vorhand', case=False, na=False)].index.tolist()
            label_rückhand = df[df['label'].str.contains('rückhand', case=False, na=False)].index.tolist()
            label_schmetterball = df[df['label'].str.contains('schmetterball', case=False, na=False)].index.tolist()
            label_kein_schlag = df[df['label'].notna()].index

            numpy_set_vorhand = extract_data(label=label_vorhand,last_index=last_index,data=data,numpy_set=numpy_set_vorhand,herz=herz)
            numpy_set_rückhand = extract_data(label=label_rückhand,last_index=last_index,data=data,numpy_set=numpy_set_rückhand,herz=herz)
            numpy_set_schmetterball = extract_data(label=label_schmetterball,last_index=last_index,data=data,numpy_set=numpy_set_schmetterball,herz=herz)

            max_label = max(len(label_vorhand) , len(label_rückhand) , len(label_schmetterball))
            numpy_set_kein_schlag = extract_data_no_hit(label=label_kein_schlag,last_index=last_index,data=data,numpy_set=numpy_set_kein_schlag,max_label=max_label,herz=herz)

    return numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball, numpy_set_kein_schlag

def read_data(data_path_30,data_path_90,data_path_100):
    numpy_set_vorhand = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_rückhand = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_schmetterball = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_kein_schlag = np.empty((0, hit_duration + 1, feature), dtype=float)

    numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball, numpy_set_kein_schlag = get_csv_from_directory(data_path_30,30,numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball,numpy_set_kein_schlag)
    numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball, numpy_set_kein_schlag = get_csv_from_directory(data_path_90,90,numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball,numpy_set_kein_schlag)
    numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball, numpy_set_kein_schlag = get_csv_from_directory(data_path_100,100,numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball,numpy_set_kein_schlag)

    numpy_set_kein_schlag = numpy_set_kein_schlag[:600, :, :]

    labels_vorhand = np.zeros(numpy_set_vorhand.shape[0], dtype=int)      # Klasse 0 für 'vorhand'
    labels_rückhand = np.ones(numpy_set_rückhand.shape[0], dtype=int)    # Klasse 1 für 'rückhand'
    labels_schmetterball = np.full(numpy_set_schmetterball.shape[0], 2)   # Klasse 2 für 'schmetterball'
    labels_kein_schlag = np.full(numpy_set_kein_schlag.shape[0], 3)   # Klasse 3 für 'kein_schlag'

    y = np.concatenate([labels_vorhand, labels_rückhand, labels_schmetterball ,labels_kein_schlag])
    X = np.concatenate([numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball,numpy_set_kein_schlag])

    #print(numpy_set_vorhand.shape)
    #print(numpy_set_rückhand.shape)
    #print(numpy_set_schmetterball.shape)
    #print(numpy_set_kein_schlag.shape)

    return X,y


def create_model(hit_duration, optimizer):
    model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(hit_duration+1,feature)),
            keras.layers.SimpleRNN(units=39, activation='relu', return_sequences=True),
            keras.layers.SimpleRNN(units=39, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = 4  # Vorhand, Rückhand, schmetterball, kein Schlag

data_path_30 = '../labeled_data_raw_30_herz/' #Ordner in dehm die roh daten liegen
data_path_90 = '../labeled_data_raw_90_herz/'
data_path_100 = '../labeled_data_raw_100_herz/'

feature_list = ['accelerometerAccelerationX(G)','accelerometerAccelerationY(G)','accelerometerAccelerationZ(G)']#,'motionYaw(rad)','motionRoll(rad)','motionPitch(rad)']
feature = len(feature_list)

LABELS = ['vorhand', 'rückhand', 'schmetterball','kein_schlag']

param_grid = {
    'optimizer': ['adam', 'rmsprop','sgd','adagrad'],
    'epochs': [60],#, 40, 50, 60, 70],
    'batch_size': [55,56,57,58, 60, 61,62],#, 40, 50, 60, 70],
    'hit_duration' : [32,34,36,38,40,42],
}

hit_duration = None

def main():
    start_time = time.time()
    best_accuracy = 0.0
    best_f1 = 0.0
    best_params = {}

    parameter_grid = list(ParameterGrid(param_grid))
    total_iterations = len(parameter_grid)

    progress_bar = tqdm(total=total_iterations, position=0, leave=True)

    for params in ParameterGrid(param_grid):
        global hit_duration
        hit_duration = params['hit_duration']

        X, y = read_data(data_path_30, data_path_90, data_path_100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = create_model(params['hit_duration'], optimizer=params['optimizer'])
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
        f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average='weighted')
        
        #if accuracy > best_accuracy:
        #    best_accuracy = accuracy
        #    best_params = params

        if f1 > best_f1:
            best_f1 = f1
            best_params = params

        progress_bar.update(1)
    
    end_time = time.time()
    progress_bar.close()
    print(f"Beste Parameter: {best_params}")
    print(f"Bester F1: {best_f1}")
    print(f"Zeit {end_time-start_time}")

main()

#print(best_hit_duration)
#print(best_f1_score)

#print(f1_scores)
#print(hit_durations_res)
#print(batch_sizes_res)

# Linienplot erstellen
#plt.figure(figsize=(10, 6))
#plt.plot(hit_durations_res, f1_scores, marker='o', linestyle='-', color='b')#

#plt.xlabel('Hit-duration')
#plt.ylabel('F1-Score')

# Gitternetz hinzufügen
#plt.grid(True)

# Plot anzeigen
#plt.show()

