import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

plt.switch_backend('TkAgg')

data = pd.read_csv("C:/Users/henry/PycharmProjects/ML-Projekt-Tischtennis4/labeling/daten/2024-05-28_12_28_58_FUNK - Kopie.csv")

fig = px.line(data, y='accelerometerAccelerationX(G)', title='Accelerometer Acceleration X (G)')
fig.update_layout(xaxis_title='Index', yaxis_title='Accelerometer Acceleration X (G)')

# Anzeige des Plots
fig.show()