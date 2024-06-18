import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("stream Apple Watch 240617 17_31_45.csv")

print(data['accelerometerAccelerationX(G)'])

plt.plot(data['accelerometerAccelerationX(G)']) 

plt.xlabel('Index') 
plt.show()