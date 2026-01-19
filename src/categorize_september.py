import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Carica il dataset (sostituisci 'nome_file.csv' con il percorso del tuo file)
df = pd.read_csv('./data/updated_data.csv')

# Filtra i dati per il mese di settembre (colonna 'MonthDeparture')
df_settembre = df[df['MonthDeparture'] == 9]

# Funzione per categorizzare LeadTime_Days in settimane
print(df_settembre)
def categorizza_settimane(days):
    if pd.isna(days):
        return 'N/A'
    settimane = math.ceil(days / 7)
    return f'{settimane} settimane'

df_settembre['LeadTime_Settimane'] = df_settembre['LeadTime_Days'].apply(categorizza_settimane)

# Calcola il tasso di cancellazione per ogni categoria di LeadTime_Settimane
tasso_cancellazione = df_settembre.groupby('LeadTime_Settimane')['Cancel'].agg(
    tasso_cancellazione=('mean')
).reset_index()

# Ordina per tasso di cancellazione decrescente
tasso_cancellazione = tasso_cancellazione.sort_values(by='tasso_cancellazione', ascending=False)

# Visualizza il risultato
print(tasso_cancellazione)

# Grafico a barre per visualizzare il tasso di cancellazione
plt.figure(figsize=(10, 6))
sns.barplot(
    x='LeadTime_Settimane',
    y='tasso_cancellazione',
    data=tasso_cancellazione,
    palette='viridis'
)
plt.title('Tasso di cancellazione per Lead Time (Settimane) - Settembre')
plt.xlabel('Lead Time (Settimane)')
plt.ylabel('Tasso di Cancellazione')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
