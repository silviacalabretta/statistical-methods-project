import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# 1. Caricamento e pulizia nomi colonne
data = pd.read_csv("./data/updated_data.csv")
data.columns = data.columns.str.strip()

# --- CAMBIO CHIAVE: Da "To" a "From" ---
FromOrTo = "From" 

# 2. Pre-calcolo delle maschere booleane per i veicoli
is_train = data['VehicleClass'] == 2
is_plane = (data['VehicleClass'] != 2) & (data['Vehicle_Plane'] == True)
is_int_plane = (data['VehicleClass'] != 2) & (data['Vehicle_IntPlane'] == True)
is_bus = (data['VehicleClass'] != 2) & (data['Vehicle_Plane'] == False) & (data['Vehicle_IntPlane'] == False)

# 3. Creazione del DataFrame riassuntivo per città (PARTENZE)
byCity = data.groupby(FromOrTo).agg(
    CancellationRate=('Cancel', 'mean'),
    NumberOfCancellation=('Cancel', 'sum'),
    DomesticDepRate=('Domestic', 'mean'), # Cambiato nome in DepRate (Departures)
    meanCost=('Price', 'mean'),
    TotalDepartures=(FromOrTo, 'count')
)

# 4. Aggiunta dei conteggi specifici per tipo di veicolo (PARTENZE)
byCity['numberOfTrainDeparted'] = data[is_train].groupby(FromOrTo)[FromOrTo].count()
byCity['numberOfPlanesDeparted'] = data[is_plane].groupby(FromOrTo)[FromOrTo].count()
byCity['numberOfIntPlanesDeparted'] = data[is_int_plane].groupby(FromOrTo)[FromOrTo].count()
byCity['numberOfBusDeparted'] = data[is_bus].groupby(FromOrTo)[FromOrTo].count()

# 5. Pulizia finale e reset index
byCity = byCity.fillna(0).reset_index()

# --- FILTRO SOGLIA MINIMA ---
min_trips_threshold = 20 
byCity_filtered = byCity[byCity['TotalDepartures'] >= min_trips_threshold].copy()

print(f"Analisi PARTENZE - Città analizzate: {len(byCity_filtered)} (escluse {len(byCity) - len(byCity_filtered)} sotto soglia)")

# 6. Preparazione Grafici
numeric_cols = byCity_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'CancellationRate' in numeric_cols:
    numeric_cols.remove('CancellationRate')
if FromOrTo in numeric_cols:
    numeric_cols.remove(FromOrTo)

n_cols = 3
n_rows = math.ceil(len(numeric_cols) / n_cols)

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 6 * n_rows))
fig.suptitle(f"Analisi Cancellazioni in PARTENZA (Minimo {min_trips_threshold} viaggi)", fontsize=20, y=1.02)

axes = axes.flatten()

# Lista colonne per scala logaritmica (aggiornata con i nuovi nomi)
cols_to_log = ['TotalDepartures', 'numberOfTrainDeparted', 'numberOfPlanesDeparted', 
               'numberOfIntPlanesDeparted', 'numberOfBusDeparted', 'NumberOfCancellation']

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.scatterplot(data=byCity_filtered, x=col, y='CancellationRate', alpha=0.6, ax=ax)
    
    if col in cols_to_log:
        ax.set_xscale('log')
        ax.set_xlabel(f"{col} (Scala Logaritmica)")
    else:
        ax.set_xlabel(col)

    ax.set_title(f"Rate vs {col}")
    ax.set_ylabel("Cancellation Rate")
    ax.grid(True, which="both", ls="-", alpha=0.2)

# Nascondi subplot vuoti
for k in range(i + 1, len(axes)):
    axes[k].axis('off')

plt.tight_layout()
plt.savefig("analisi_cancellazioni_filtrata_partenze.png", bbox_inches='tight')
plt.show()