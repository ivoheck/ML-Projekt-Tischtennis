import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def extract_data(label,last_index,data,herz):
    for index in label:
        if herz == 30:
            if index - window_size/2 >= 0 and index + window_size/2 <= last_index:
                data_set = data[index-int(window_size/2):index+int(window_size/2)+1]

        elif herz == 90:
            if index - (window_size/2)*3 >= 0 and index + (window_size/2)*3 <= last_index:
                data_set = data[index-int(window_size/2)*3:index+int(window_size/2)*3+3]

                data_set = data_set.groupby(np.arange(len(data_set)) // 3).mean()
        
        #TODO: find better way for doing this
        elif herz == 100:
            if index - (window_size/2)*3 >= 0 and index + (window_size/2)*3 <= last_index:
                data_set = data[index-int(window_size/2)*3:index+int(window_size/2)*3+3]

                data_set = data_set.groupby(np.arange(len(data_set)) // 3).mean()

        else:
            print(herz, 'herz is not supported')
            return
        
        max_acceleration_x.append(data_set['accelerometerAccelerationX(G)'].max())
        max_acceleration_y.append(data_set['accelerometerAccelerationY(G)'].max())
        max_acceleration_z.append(data_set['accelerometerAccelerationZ(G)'].max())

        diff_acceleration_x.append(abs(data_set['accelerometerAccelerationX(G)'].max() - data_set['accelerometerAccelerationX(G)'].min()))
        diff_acceleration_y.append(abs(data_set['accelerometerAccelerationY(G)'].max() - data_set['accelerometerAccelerationY(G)'].min()))
        diff_acceleration_z.append(abs(data_set['accelerometerAccelerationZ(G)'].max() - data_set['accelerometerAccelerationZ(G)'].min()))

        min_acceleration_x.append(data_set['accelerometerAccelerationX(G)'].min())
        min_acceleration_y.append(data_set['accelerometerAccelerationY(G)'].min())
        min_acceleration_z.append(data_set['accelerometerAccelerationZ(G)'].min())

def extract_data_no_hit(label,last_index,data,herz):
    #index is the index of the label in the data set, i is the index of the list
    count = 0
    for i,index in enumerate(label):
        #TODO: handel first case
        if i == 0:
            pass

        elif i + 1 <= len(label) -1:
            prev = index
            
            while label[i+1] - prev >= window_size + 1:
                count += 1
                prev += int(window_size) + int(window_size/2) + 1

                data_set = data[prev + int(window_size/2) : prev+int(window_size) + int(window_size/2) + 1]
                max_acceleration_x_no_hit.append(data_set['accelerometerAccelerationX(G)'].max())
                max_acceleration_y_no_hit.append(data_set['accelerometerAccelerationY(G)'].max())
                max_acceleration_z_no_hit.append(data_set['accelerometerAccelerationZ(G)'].max())

                min_acceleration_x_no_hit.append(data_set['accelerometerAccelerationX(G)'].min())
                min_acceleration_y_no_hit.append(data_set['accelerometerAccelerationY(G)'].min())
                min_acceleration_z_no_hit.append(data_set['accelerometerAccelerationZ(G)'].min())

                diff_acceleration_x_no_hit.append(abs(data_set['accelerometerAccelerationX(G)'].max() - data_set['accelerometerAccelerationX(G)'].min()))
                diff_acceleration_y_no_hit.append(abs(data_set['accelerometerAccelerationY(G)'].max() - data_set['accelerometerAccelerationY(G)'].min()))
                diff_acceleration_z_no_hit.append(abs(data_set['accelerometerAccelerationZ(G)'].max() - data_set['accelerometerAccelerationZ(G)'].min()))
                

def get_csv_from_directory(data_path,herz):
    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        if os.path.isfile(path):  # Überprüfen, ob es eine Datei ist (kein Unterordner)

            df = pd.read_csv(path)
            data = df[feature_list]
            label = df['label']

            last_index = label.shape[0] - 1

            label_vorhand_data_points = df[df['label'].str.contains('vorhand', case=False, na=False)].index.tolist()
            label_rückhand_data_points = df[df['label'].str.contains('rückhand', case=False, na=False)].index.tolist()
            label_schmetterball_data_points = df[df['label'].str.contains('schmetterball', case=False, na=False)].index.tolist()

            label_kein_schlag_data_points = df[df['label'].notna()].index

            extract_data(label=label_vorhand_data_points,last_index=last_index,data=data,herz=herz)
            extract_data(label=label_rückhand_data_points,last_index=last_index,data=data,herz=herz)
            extract_data(label=label_schmetterball_data_points,last_index=last_index,data=data,herz=herz)

            extract_data_no_hit(label=label_kein_schlag_data_points,last_index=last_index,data=data,herz=herz)


data_path_30 = '../labeled_data_raw_30_herz/' #Ordner in dehm die roh daten liegen
data_path_90 = '../labeled_data_raw_90_herz/'
data_path_100 = '../labeled_data_raw_100_herz/'

feature_list = ['accelerometerAccelerationX(G)','accelerometerAccelerationY(G)','accelerometerAccelerationZ(G)']

label_vorhand, label_rückhand, label_schmetterball, label_angabe_vorhand, label_angabe_rückhand = 0,0,0,0,0
window_size = 10 #(immer plus 1)

max_acceleration_x = []
max_acceleration_y = []
max_acceleration_z = []

diff_acceleration_x = []
diff_acceleration_y = []
diff_acceleration_z = []

min_acceleration_x = []
min_acceleration_y = []
min_acceleration_z = []

max_acceleration_x_no_hit = []
max_acceleration_y_no_hit = []
max_acceleration_z_no_hit = []

diff_acceleration_x_no_hit = []
diff_acceleration_y_no_hit = []
diff_acceleration_z_no_hit = []

min_acceleration_x_no_hit = []
min_acceleration_y_no_hit = []
min_acceleration_z_no_hit = []

def main():
    get_csv_from_directory(data_path_30,30)
    get_csv_from_directory(data_path_90,90)
    get_csv_from_directory(data_path_100,100)

    print('max acc[xyz]',max(max_acceleration_x),max(max_acceleration_y),max(max_acceleration_z))
    print('min acc[xyz]',min(min_acceleration_x),min(min_acceleration_y),min(min_acceleration_z))

    print('max acc[xyz] no hit',max(max_acceleration_x_no_hit),max(max_acceleration_y_no_hit),max(max_acceleration_z_no_hit))
    print('min acc[xyz] no hit',min(min_acceleration_x_no_hit),min(min_acceleration_y_no_hit),min(min_acceleration_z_no_hit))

    print('avg max acc [xyz]',sum(max_acceleration_x)/len(max_acceleration_x),sum(max_acceleration_y)/len(max_acceleration_y),sum(max_acceleration_z)/len(max_acceleration_z))
    print('avg min acc [xyz]',sum(min_acceleration_x)/len(min_acceleration_x),sum(min_acceleration_y)/len(min_acceleration_y),sum(min_acceleration_z)/len(min_acceleration_z))

    print('avg max acc [xyz] no hit',sum(max_acceleration_x_no_hit)/len(max_acceleration_x_no_hit),sum(max_acceleration_y_no_hit)/len(max_acceleration_y_no_hit),sum(max_acceleration_z_no_hit)/len(max_acceleration_z_no_hit))
    print('avg max acc [xyz] no hit',sum(min_acceleration_x_no_hit)/len(min_acceleration_x_no_hit),sum(min_acceleration_y_no_hit)/len(min_acceleration_y_no_hit),sum(min_acceleration_z_no_hit)/len(min_acceleration_z_no_hit))

    #print(min(max_acceleration_x),min(max_acceleration_y),min(max_acceleration_z))
    #print(max(min_acceleration_x),max(min_acceleration_y),max(min_acceleration_z))
    print('avg diff [x,y,z]',sum(diff_acceleration_x)/len(diff_acceleration_x),sum(diff_acceleration_y)/len(diff_acceleration_y),sum(diff_acceleration_z)/len(diff_acceleration_z))
    print('avg diff [x,y,z] no hit',sum(diff_acceleration_x_no_hit)/len(diff_acceleration_x_no_hit),sum(diff_acceleration_y_no_hit)/len(diff_acceleration_y_no_hit),sum(diff_acceleration_z_no_hit)/len(diff_acceleration_z_no_hit))

if __name__ == '__main__':
    main()