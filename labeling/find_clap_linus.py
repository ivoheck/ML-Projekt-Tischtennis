import pandas as pd
import plotly.graph_objects as go

# Daten laden
data = pd.read_csv(r"C:\Uni\\Watch_L_001_2006.csv")

# Erstellen der interaktiven Plotly-Graphen
fig = go.Figure()

# Hinzufügen der Linien mit Markierungen
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['accelerometerAccelerationX(G)'],
    mode='lines+markers',
    name='Accelerometer Acceleration X (G)',
    marker=dict(size=8, symbol='circle', color='blue'),  # Größe und Farbe der Marker festlegen
    text=[f'Index: {i}<br>Acceleration X: {val:.2f} G' for i, val in enumerate(data['accelerometerAccelerationX(G)'])],
    hoverinfo='text'  # Hover-Info anpassen
))

# Layout anpassen
fig.update_layout(
    title='Accelerometer Acceleration X (G)',
    xaxis_title='Index',
    yaxis_title='Accelerometer Acceleration X (G)',
    hovermode='closest',  # Hovermodus auf "closest" setzen, um die Tooltip-Informationen dem Mauszeiger folgen zu lassen
    font=dict(size=18),  # Schriftgröße für das gesamte Layout anpassen
    margin=dict(l=0, r=0, t=40, b=20),  # Layout-Margen anpassen
    hoverlabel=dict(
        font_size=200,  # Schriftgröße der Tooltip-Labels anpassen
        font_family="Arial"
    )
)

# Anzeige des Plots
fig.show()
