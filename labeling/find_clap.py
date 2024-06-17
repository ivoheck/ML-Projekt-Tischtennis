import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("daten/TischtennisTest1.csv", usecols=[2])

plt.plot(data) 

plt.xlabel('Index') 
plt.show()