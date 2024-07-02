import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier
import os
import random
import torch

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
            
            while label[i+1] - prev >= hit_duration + 1:
                count += 1
                prev += int(hit_duration) + int(hit_duration/2) + 1

                data_set = data[prev + int(hit_duration/2) : prev+int(hit_duration) + int(hit_duration/2) + 1]
                numpy_data_set = data_set.values
                numpy_set = np.append(numpy_set, [numpy_data_set], axis=0).astype(np.float32)

                if count >= max_label:
                   return numpy_set

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
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #return X_train, X_test, y_train, y_test

    return X,y

num_classes = 4  # Vorhand, Rückhand, schmetterball, kein Schlag
hit_duration = 64 # datenpunkte für einen schlag bei 30 insgesamt ein datenpunkt mehr als angegeben

data_path_30 = '../labeled_data_raw_30_herz/' #Ordner in dehm die roh daten liegen
data_path_90 = '../labeled_data_raw_90_herz/'
data_path_100 = '../labeled_data_raw_100_herz/'

feature_list = ['accelerometerAccelerationX(G)','accelerometerAccelerationY(G)','accelerometerAccelerationZ(G)']#,'motionYaw(rad)','motionRoll(rad)','motionPitch(rad)']
feature = len(feature_list)

# Set parameters for data splitting and training
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 50
LABELS = ['vorhand', 'rückhand', 'schmetterball','kein_schlag']

#X_train, X_test, y_train, y_test = read_data(data_path_30,data_path_90,data_path_100)
X, y = read_data(data_path_30,data_path_90,data_path_100)

# Encode the labels using One-Hot-Encoding
#y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
#y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)


# StratifiedKFold für Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_f1 = 0.0
best_conf_matrix = None
for train_index, val_index in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    y_train_fold_encoded = tf.keras.utils.to_categorical(y_train_fold, num_classes=num_classes)
    y_val_fold_encoded = tf.keras.utils.to_categorical(y_val_fold, num_classes=num_classes)

    #Set up model
    model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(hit_duration + 1,feature)),
    keras.layers.SimpleRNN(units=64, activation='relu', return_sequences=True),
    keras.layers.SimpleRNN(units=64, activation='relu'),
    keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    #print(model.summary())

    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #history = model.fit(x=X_train_fold, y=y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stopping], verbose=0)
    history = model.fit(x=X_train_fold, y=y_train_fold_encoded, validation_data=(X_val_fold, y_val_fold_encoded), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stopping], verbose=0)

    y_pred = np.argmax(model.predict(x=X_val_fold), axis=1)
    conf_matrix = confusion_matrix(y_val_fold, y_pred)

    # F1-Score und andere Metriken berechnen
    report = classification_report(y_val_fold, y_pred, target_names=LABELS, output_dict=True)

    # Extrahiere den gewichteten F1-Score
    weighted_f1_score = report['weighted avg']['f1-score']
    print(f"Weighted F1-Score: {weighted_f1_score:.4f}")

    if weighted_f1_score > best_f1:
        best_f1 = weighted_f1_score
        best_model = model
        best_conf_matrix = conf_matrix


fig = plt.figure()
print(best_f1)
sns.heatmap(best_conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# Speichern des gesamten Modells
best_model.save('model_cross_val.keras')