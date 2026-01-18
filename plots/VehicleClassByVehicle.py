import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    

data=pd.read_csv("./data/tickets_data.csv")
#Created,CancelTime,DepartureTime,BillID,TicketID,ReserveStatus,UserID,Male,Price,CouponDiscount,From,To,Domestic,VehicleType,VehicleClass,TripReason,Vehicle,Cancel,HashPassportNumber_p,HashEmail,BuyerMobile,NationalCode

bus_data=data[data["Vehicle"]=="Bus"][["VehicleClass","Vehicle","TripReason"]]
plane_data=data[data["Vehicle"]=="Plane"][["VehicleClass","Vehicle","TripReason"]]
train_data=data[data["Vehicle"]=="Train"][["VehicleClass","Vehicle","TripReason"]]
internplane_data=data[data["Vehicle"]=="InternationalPlane"][["VehicleClass","Vehicle","TripReason"]]

print(train_data)
# Supponiamo che i tuoi dataframe siano già definiti come:
# bus_data, plane_data, train_data

# Funzione per creare un barplot da un dataframe

# Chiamata alla funzione per ogni dataframe
plot_bar_from_dataframe(bus_data, 'Bus')
plot_bar_from_dataframe(plane_data, 'Plane')
plot_bar_from_dataframe(train_data, 'Train')
plot_bar_from_dataframe(internplane_data, 'InternationalPlane')
