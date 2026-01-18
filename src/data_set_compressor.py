import pandas as pd

# 1. Caricamento e pulizia nomi colonne
data = pd.read_csv("./data/updated_data.csv")
data.columns = data.columns.str.strip()

FromOrTo="To" #change to from
# 2. Pre-calcolo delle maschere booleane per i veicoli
# (Rende il codice molto più leggibile rispetto a filtri complessi nel groupby)
is_train = data['VehicleClass'] == 2
is_plane = (data['VehicleClass'] != 2) & (data['Vehicle_Plane'] == True)
is_int_plane = (data['VehicleClass'] != 2) & (data['Vehicle_IntPlane'] == True)
is_bus = (data['VehicleClass'] != 2) & (data['Vehicle_Plane'] == False) & (data['Vehicle_IntPlane'] == False)

# 3. Creazione del DataFrame riassuntivo per città
# Iniziamo con le metriche calcolabili direttamente tramite aggregazione
byCity = data.groupby(FromOrTo).agg(
    CancellationRate=('Cancel', 'mean'),
    NumberOfCancellation=('Cancel', 'sum'),
    DomesticDepRate=('Domestic', 'mean'),
    meanCost=('Price', 'mean'),
    TotalDepartures=(FromOrTo, 'count')
)

# 4. Aggiunta dei conteggi specifici per tipo di veicolo
# Usiamo il groupby sulla colonna FromOrTo applicato alle maschere create sopra
byCity['numberOfTrainDeparted'] = data[is_train].groupby(FromOrTo)[FromOrTo].count()
byCity['numberOfPlanesDeparted'] = data[is_plane].groupby(FromOrTo)[FromOrTo].count()
byCity['numberOfInternationalPlanesDeparted'] = data[is_int_plane].groupby(FromOrTo)[FromOrTo].count()
byCity['numberOfBusDeparted'] = data[is_bus].groupby(FromOrTo)[FromOrTo].count()

# 5. Pulizia finale
# Riempiamo i valori NaN con 0 (es: città che non hanno treni avrebbero NaN)
byCity = byCity.fillna(0)

# Trasformiamo l'indice (la città) in una colonna normale
byCity = byCity.reset_index()

# 6. Output
print("Prime 5 righe del DataFrame elaborato:")
print(byCity.head())

# Opzionale: salvataggio
byCity.to_csv("./data/city_arrival_summary.csv", index=False)