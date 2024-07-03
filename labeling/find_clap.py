import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("daten/Watch_L_006_2006.csv")

print(data['accelerometerAccelerationX(G)'])

plt.plot(data['accelerometerAccelerationX(G)']) 

plt.xlabel('Index') 
plt.show()