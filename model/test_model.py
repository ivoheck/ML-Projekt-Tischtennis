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
        
def hadle_data(predictions,axs):
    count_vorhand = 0
    count_rückhand = 0
    count_schmetterball = 0

    vorhand = []
    rückhand = []
    schmetterball = []

    index = 0
    i = 0
    last = [None,None,None,None,None]
    overshoot = 1 #15-20
    while i <= len(predictions) -1:
        data_set = predictions[i]
        if data_set == 0:#vorhand
            if all(last[-i] == 0 for i in range(1, 3)):#2-5
                count_vorhand += 1
                axs[0, 0].axvline(x=index+(num_rows/2), color='blue', linestyle='--', ymax=0.5)
                vorhand.append(index+(num_rows/2))
                i += overshoot
                index += overshoot * step_size
                last = [3]

            i += 1
            index += 1 * step_size
            last.append(0)

        elif data_set == 1:
            if all(last[-i] == 1 for i in range(1, 3)):
                count_rückhand += 1
                axs[0, 1].axvline(x=index+(num_rows/2), color='blue', linestyle='--', ymax=0.5)
                rückhand.append(index+(num_rows/2))
                i += overshoot
                index += overshoot * step_size
                last = [3]

            i += 1
            index += 1 * step_size
            last.append(1)

        elif data_set == 2:
            if all(last[-i] == 2 for i in range(1, 3)):
                count_schmetterball += 1
                axs[1, 0].axvline(x=index+num_rows/2, color='blue', linestyle='--', ymax=0.5)
                schmetterball.append(index+(num_rows/2))
                i += overshoot
                index += overshoot * step_size
                last = [3]

            i += 1
            index += 1 * step_size
            last.append(2)

        else:
            i += 1
            index += 1 * step_size
            #plt.axvline(x=index, color='yellow', linestyle='--', ymax=0.5)
            last = [3]

    return count_vorhand, count_rückhand, count_schmetterball, vorhand, rückhand, schmetterball

def merge_indices(indices, threshold):
    if not indices:
        return []

    merged_indices = []
    current_group = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] - current_group[-1] <= threshold:
            current_group.append(indices[i])
        else:
            merged_indices.append(sum(current_group) // len(current_group))
            current_group = [indices[i]]

    merged_indices.append(sum(current_group) // len(current_group))
    return merged_indices

def clean_data(vorhand, rückhand, schmetterball, axs):
    vorhand = merge_indices(vorhand,merge_treshold)
    rückhand = merge_indices(rückhand,merge_treshold)
    schmetterball = merge_indices(schmetterball,merge_treshold)

    for i in vorhand:
        axs[0, 0].axvline(x=i, color='blue', linestyle='--', ymax=0.5)

    for i in rückhand:
        axs[0, 1].axvline(x=i, color='blue', linestyle='--', ymax=0.5)

    for i in schmetterball:
        axs[1, 0].axvline(x=i, color='blue', linestyle='--', ymax=0.5)

    print(len(vorhand))

    return vorhand,rückhand,schmetterball,len(vorhand),len(rückhand),len(schmetterball)

def init_plot(df):
    fig, axs = plt.subplots(2, 2)  
    axs[0, 0].plot(df['accelerometerAccelerationX(G)'])
    axs[0, 0].set_title('Vorhand')

    axs[0, 1].plot(df['accelerometerAccelerationX(G)'])
    axs[0, 1].set_title('Rückhand')

    axs[1, 0].plot(df['accelerometerAccelerationX(G)'])
    axs[1, 0].set_title('Schmetterball')

    return axs

def plot_plot(axs,df,count_vorhand, count_rückhand, count_schmetterball):
    true_lable = df['label'].tolist()

    count_vorhand_true = 0
    count_rückhand_true = 0
    count_schmetterball_true = 0

    count_angabe_rückhand_true = 0
    count_angabe_vorhand_true = 0

    for i,lable in enumerate(true_lable):
        if lable == 'vorhand':
            count_vorhand_true +=1
            axs[0, 0].axvline(x=i, color='green', linestyle='--', ymin=0.5)

        elif lable == 'angabe_vorhand':
            count_angabe_vorhand_true += 1
            axs[0, 0].axvline(x=i, color='red', linestyle='--', ymin=0.5)
        
        elif lable == 'rückhand':
           count_rückhand_true += 1
           axs[0, 1].axvline(x=i, color='green', linestyle='--', ymin=0.5)

        elif lable == 'angabe_rückhand':
            count_angabe_rückhand_true += 1
            axs[0, 1].axvline(x=i, color='red', linestyle='--', ymin=0.5)

        elif lable == 'schmetterball':
            count_schmetterball_true += 1
            axs[1, 0].axvline(x=i, color='green', linestyle='--', ymin=0.5)

    data = [
        ['Labeled', f'{count_vorhand_true} ({count_vorhand_true + count_angabe_vorhand_true})', f'{count_rückhand_true} ({count_rückhand_true + count_angabe_rückhand_true})', count_schmetterball_true],
        ['Predicted', count_vorhand, count_rückhand, count_schmetterball]
    ]

    columns = ['Parameter', 'Vorhand (Mit V. Angaben)', 'Rückhand (Mit R. Angaben)', 'Schmetterball']

    axs[1, 1].axis('tight')
    axs[1, 1].axis('off')
    table = axs[1, 1].table(cellText=data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.xlabel('Index')
    plt.tight_layout()
    plt.show()


def plot_data(df,predictions_df):
    
    predictions = predictions_df['Prediction'].tolist()

    axs = init_plot(df)
    count_vorhand, count_rückhand, count_schmetterball, vorhand, rückhand, schmetterball = hadle_data(predictions,axs)
    plot_plot(axs,df,count_vorhand, count_rückhand, count_schmetterball)

    #plot_odds(predictions)
    
    axs = init_plot(df=df)
    vorhand, rückhand, schmetterball, count_vorhand, count_rückhand, count_schmetterball = clean_data(vorhand, rückhand, schmetterball,axs)
    plot_plot(axs,df,count_vorhand, count_rückhand, count_schmetterball)

data_path = 'test_data'
num_rows = 64 * 3 #für 90 herz daten
step_size = 6
batch_size = 5
distance_threshold= batch_size * step_size
feature_list = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)']
merge_treshold = 50

extracts = []

#Funktioniert für 90 herz daten
def main():
    df = get_data(data_path)

    for start in range(0, len(df) - num_rows + 1, step_size):
        extract = df.iloc[start:start + num_rows][feature_list].values
        block_avg = pd.DataFrame(extract).groupby(np.arange(len(extract)) // 3).mean().values
        extracts.append(block_avg)

    X_test = np.array(extracts)

    print(X_test.shape) 

    # Laden des gesamten Modells
    loaded_model = keras.models.load_model('model.h5')

    loaded_model.summary()

    y_predicted = np.argmax(loaded_model.predict(x=X_test), axis=1)

    predictions_df = pd.DataFrame(y_predicted, columns=['Prediction'])

    plot_data(df,predictions_df)

if __name__ == '__main__':
    main()