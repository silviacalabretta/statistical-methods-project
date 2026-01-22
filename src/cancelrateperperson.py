
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('./data/old_modified_data.csv')

summary = df.groupby('NationalCode').agg(
    Total_Tickets=('Cancel', 'count'),
    Cancelled_Tickets=('Cancel', 'sum'),
    Cancel_Rate=('Cancel', 'mean'),
    Bus_Tickets=('Vehicle_Bus', 'sum'),
    Plane_Tickets=('Vehicle_Plane', 'sum'),
    Train_Tickets=('Vehicle_Train', 'sum'),
).reset_index()

# Calcolo delle cancellazioni per mezzo di trasporto
cancel_by_vehicle = df[df['Cancel'] == 1].groupby('NationalCode').agg(
    Cancelled_Bus=('Vehicle_Bus', 'sum'),
    Cancelled_Plane=('Vehicle_Plane', 'sum'),
    Cancelled_Train=('Vehicle_Train', 'sum'),
).reset_index()

# Unione delle informazioni
final_summary = pd.merge(summary, cancel_by_vehicle, on='NationalCode', how='left').fillna(0)

# # Calcoliamo il tasso di cancellazione per ogni mezzo di trasporto per ogni utente
# def calculate_cancel_rate_by_vehicle(df):
#     vehicles = ['Bus', 'Plane', 'Train']
#     cancel_rates = {}
    
#     for vehicle in vehicles:
#         tickets_col = f'{vehicle}_Tickets'
#         cancel_col = f'Cancelled_{vehicle}'
        
#         # Numero di biglietti acquistati per mezzo
#         total_tickets = df[tickets_col].sum()
#         # Numero di biglietti cancellati per mezzo
#         cancelled_tickets = df[cancel_col].sum()
        
#         # Tasso di cancellazione per mezzo
#         cancel_rate = cancelled_tickets / total_tickets if total_tickets > 0 else 0
#         cancel_rates[vehicle] = cancel_rate
    
#     return cancel_rates

# Calcoliamo i tassi di cancellazione per ogni mezzo
#cancel_rates = calculate_cancel_rate_by_vehicle(final_summary)

# Dati per il plot
# vehicles = ['Bus', 'Plane', 'Train']
# cancel_rates = list(cancel_rates.values())

# Creazione del plot

# # Plottiamo le barre per ogni mezzo
# bars = ax.bar(index, cancel_rates, bar_width, color=['blue', 'orange', 'green'])

# # Configuriamo il plot
# ax.set_xlabel('Mezzo di Trasporto')
# ax.set_ylabel('Tasso di Cancellazione')
# ax.set_title('Tasso di Cancellazione per Mezzo di Trasporto')
# ax.set_xticks(index)
# ax.set_xticklabels(vehicles)

# # Aggiungiamo i valori sopra le barre
# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

# plt.tight_layout()
# plt.show()
reduced = final_summary[final_summary["Total_Tickets"] > 5]

# Creiamo il grafico a dispersione
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot con stili migliorati
scatter = ax.scatter(
    reduced["Total_Tickets"], 
    reduced["Cancel_Rate"], 
    c='darkblue',  # Colore dei punti
    s=100,         # Dimensione dei punti
    alpha=0.7,     # Trasparenza
    edgecolors='black'  # Bordo dei punti
)
from tabulate import tabulate
a=final_summary[final_summary['Cancel_Rate']==1]
print(tabulate(a[a["Total_Tickets"]>4],final_summary.columns))
print(len(a[a["Total_Tickets"]>4]))
# Aggiungiamo titoli e etichette
ax.set_title('Relationship between number of tickets and cancellation rate', fontsize=14, pad=20)
ax.set_xlabel('Total number of tickets', fontsize=12)
ax.set_ylabel('cancellation rate', fontsize=12)
plt.tight_layout()
plt.show()