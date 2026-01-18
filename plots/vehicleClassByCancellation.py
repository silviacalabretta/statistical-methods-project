import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv("./data/tickets_data.csv")


cancelled_data=data[data["Cancel"]==1][["VehicleClass","Vehicle","TripReason"]]

def plot_bar_from_dataframe(data, title):
    # Sostituisci i valori null con la stringa "null"
    data_filled = data.fillna("null")

    # Conteggio delle frequenze per ogni colonna categorica
    vehicle_class_counts = data_filled['VehicleClass'].value_counts()
    trip_reason_counts = data_filled['TripReason'].value_counts()

    # Creazione dei subplot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Barplot per VehicleClass
    bars1 = axs[0].bar(vehicle_class_counts.index, vehicle_class_counts.values, color='skyblue')
    axs[0].set_title(f'{title} - VehicleClass')
    axs[0].set_ylabel('Frequenza')

    # Aggiungi i valori sopra le barre
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, height,
                    f'{int(height)}', ha='center', va='bottom')

    # Barplot per TripReason
    bars2 = axs[1].bar(trip_reason_counts.index, trip_reason_counts.values, color='lightgreen')
    axs[1].set_title(f'{title} - TripReason')
    axs[1].set_ylabel('Frequenza')

    # Aggiungi i valori sopra le barre
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, height,
                    f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

print(cancelled_data)
plot_bar_from_dataframe(cancelled_data,"aaaa")