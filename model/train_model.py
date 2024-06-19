import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

num_classes = 3  # Vorhand, Rückhand, schmetterball, kein Schlag
hit_duration = 30 # datenpunkte für einen schlag bei 30 insgesamt ein datenpunkt mehr als angegeben
data_path = '../labeled_data_raw/' #Ordner in dehm die roh daten liegen
feature = 3

def read_data(data_path):
    numpy_set_vorhand = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_rückhand = np.empty((0, hit_duration + 1, feature), dtype=float)
    numpy_set_schmetterball = np.empty((0, hit_duration + 1, feature), dtype=float)

    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        if os.path.isfile(path):  # Überprüfen, ob es eine Datei ist (kein Unterordner)

            df = pd.read_csv(path)
            data = df[['accelerometerAccelerationX(G)','accelerometerAccelerationY(G)','accelerometerAccelerationZ(G)']]#,'motionYaw(rad)','motionRoll(rad)','motionPitch(rad)']]
            label = df['label']

            last_index = label.shape[0] - 1

            label_vorhand = df[df['label'].str.contains('vorhand', case=False, na=False)].index.tolist()
            label_rückhand = df[df['label'].str.contains('rückhand', case=False, na=False)].index.tolist()
            label_schmetterball = df[df['label'].str.contains('schmetterball', case=False, na=False)].index.tolist()

            for index in label_vorhand:
                if index - hit_duration/2 >= 0 and index + hit_duration/2 <= last_index:
                    data_set = data[index-int(hit_duration/2):index+int(hit_duration/2)+1]
                    numpy_set = data_set.values
                    numpy_set_vorhand = np.append(numpy_set_vorhand, [numpy_set], axis=0).astype(np.float32) 

            for index in label_rückhand:
                if index - hit_duration/2 >= 0 and index + hit_duration/2 <= last_index:
                    data_set = data[index-int(hit_duration/2):index+int(hit_duration/2)+1]
                    numpy_set = data_set.values
                    numpy_set_rückhand = np.append(numpy_set_rückhand, [numpy_set], axis=0).astype(np.float32) 

            for index in label_schmetterball:
                if index - hit_duration/2 >= 0 and index + hit_duration/2 <= last_index:
                    data_set = data[index-int(hit_duration/2):index+int(hit_duration/2)+1]
                    numpy_set = data_set.values
                    numpy_set_schmetterball = np.append(numpy_set_schmetterball, [numpy_set], axis=0).astype(np.float32) 

    labels_vorhand = np.zeros(numpy_set_vorhand.shape[0], dtype=int)      # Klasse 0 für 'vorhand'
    labels_rückhand = np.ones(numpy_set_rückhand.shape[0], dtype=int)    # Klasse 1 für 'rückhand'
    labels_schmetterball = np.full(numpy_set_schmetterball.shape[0], 2)   # Klasse 2 für 'schmetterball'

    y = np.concatenate([labels_vorhand, labels_rückhand, labels_schmetterball])
    X = np.concatenate([numpy_set_vorhand, numpy_set_rückhand, numpy_set_schmetterball])

    print(numpy_set_vorhand.shape)
    print(numpy_set_rückhand.shape)
    print(numpy_set_schmetterball.shape)


    #Split data in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = read_data(data_path=data_path)

#Set up model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(hit_duration + 1,feature)),
    keras.layers.SimpleRNN(units=31, activation='relu', return_sequences=True),
    keras.layers.SimpleRNN(units=31, activation='relu'),
    keras.layers.Dense(num_classes, activation=keras.activations.softmax)
])

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
print(model.summary())

# Set parameters for data splitting and training
TEST_SIZE = 0.3
BATCH_SIZE = 64
EPOCHS = 50
LABELS = ['vorhand', 'rückhand', 'schmetterball']

# Encode the labels using One-Hot-Encoding
y_train_encoded = tf.one_hot(indices=y_train, depth=3)

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