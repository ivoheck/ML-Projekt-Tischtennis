import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("stream Apple Watch 240617 17_04_36.csv")

plt.plot(data['accelerometerAccelerationX(G)']) 

plt.xlabel('Index') 
plt.show()