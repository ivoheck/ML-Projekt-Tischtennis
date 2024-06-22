import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

def extract_data(label,last_index,data,numpy_set):
    for index in label:
        if index - hit_duration/2 >= 0 and index + hit_duration/2 <= last_index:
            data_set = data[index-int(hit_duration/2):index+int(hit_duration/2)+1]
            numpy_data_set = data_set.values
            numpy_set = np.append(numpy_set, [numpy_data_set], axis=0).astype(np.float32) 

    return numpy_set

def extract_data_no_hit(label,last_index,data,numpy_set,max_label):
    #index is the index of the label in the data set, i is the index of the list
    count = 0
    for i,index in enumerate(label):
        #TODO: handel first case
        if i == 0:
            pass

        elif i + 1 <= len(label) -1:
            prev = index
            
            while label[i+1] - prev >= hit_duration + 1:
                count += 1
                prev += int(hit_duration) + int(hit_duration/2) + 1

                data_set = data[prev + int(hit_duration/2) : prev+int(hit_duration) + int(hit_duration/2) + 1]
                numpy_data_set = data_set.values
                numpy_set = np.append(numpy_set, [numpy_data_set], axis=0).astype(np.float32)

                if count >= max_label:
                   return numpy_set

    return numpy_set

def get_csv_from_directory(data_path,herz,numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball):
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

            numpy_set_vorhand = extract_data(label=label_vorhand,last_index=last_index,data=data,numpy_set=numpy_set_vorhand)
            numpy_set_rückhand = extract_data(label=label_rückhand,last_index=last_index,data=data,numpy_set=numpy_set_rückhand)
            numpy_set_schmetterball = extract_data(label=label_schmetterball,last_index=last_index,data=data,numpy_set=numpy_set_schmetterball)

            max_label = max(len(label_vorhand) , len(label_rückhand) , len(label_schmetterball))
            numpy_set_kein_schlag = extract_data_no_hit(label=label_kein_schlag,last_index=last_index,data=data,numpy_set=numpy_set_kein_schlag,max_label=max_label)

            return numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball

def read_data(data_path_30,data_path_90,data_path_100):
    numpy_set_vorhand = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_rückhand = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_schmetterball = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_kein_schlag = np.empty((0, hit_duration + 1, feature), dtype=float)

    numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball = get_csv_from_directory(data_path_30,30,numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball)

    labels_vorhand = np.zeros(numpy_set_vorhand.shape[0], dtype=int)      # Klasse 0 für 'vorhand'
    labels_rückhand = np.ones(numpy_set_rückhand.shape[0], dtype=int)    # Klasse 1 für 'rückhand'
    labels_schmetterball = np.full(numpy_set_schmetterball.shape[0], 2)   # Klasse 2 für 'schmetterball'
    labels_kein_schlag = np.full(numpy_set_kein_schlag.shape[0], 3)   # Klasse 3 für 'kein_schlag'

    y = np.concatenate([labels_vorhand, labels_rückhand, labels_schmetterball ,labels_kein_schlag])
    X = np.concatenate([numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball,numpy_set_kein_schlag])

    print(numpy_set_vorhand.shape)
    print(numpy_set_rückhand.shape)
    print(numpy_set_schmetterball.shape)
    print(numpy_set_kein_schlag.shape)


    #Split data in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


num_classes = 4  # Vorhand, Rückhand, schmetterball, kein Schlag
hit_duration = 30 # datenpunkte für einen schlag bei 30 insgesamt ein datenpunkt mehr als angegeben

data_path_30 = '../labeled_data_raw_30_herz/' #Ordner in dehm die roh daten liegen
data_path_90 = '../labeled_data_raw_90_herz/'
data_path_100 = '../labeled_data_raw_100_herz/'

feature_list = ['accelerometerAccelerationX(G)','accelerometerAccelerationY(G)','accelerometerAccelerationZ(G)','motionYaw(rad)','motionRoll(rad)','motionPitch(rad)']
feature = len(feature_list)

# Set parameters for data splitting and training
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 50
LABELS = ['vorhand', 'rückhand', 'schmetterball','kein_schlag']

X_train, X_test, y_train, y_test = read_data(data_path_30,data_path_90,data_path_100)

#Set up model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(hit_duration + 1,feature)),
    keras.layers.SimpleRNN(units=31, activation='relu', return_sequences=True),
    keras.layers.SimpleRNN(units=31, activation='relu'),
    keras.layers.Dense(num_classes, activation=keras.activations.softmax)
])

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
print(model.summary())

# Encode the labels using One-Hot-Encoding
y_train_encoded = tf.one_hot(indices=y_train, depth=num_classes)

# Train model using validation split
stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(x=X_train, y=y_train_encoded, validation_split=TEST_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[stopping])

#Model Evaluation
y_predicted = np.argmax(model.predict(x=X_test), axis=1)
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted)
fig = plt.figure()
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()