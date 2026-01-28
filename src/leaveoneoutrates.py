import pandas as pd
import numpy as np

# 1. Caricamento del dataset originale (Greggio)
df = pd.read_csv("./data/old_modified_data.csv")
print(f"Dataset caricato: {df.shape[0]} righe.")

# ==============================================================================
# FASE 1: Feature Engineering di Base (Tempo, LeadTime, Dummy)
# ==============================================================================

# 1.1 Log Lead Time (Logaritmo per normalizzare la distribuzione)
# Aggiungiamo 1 per evitare log(0)
df['LogLeadTime'] = np.log1p(df['LeadTime_Days'])

# 1.2 Time of Day (Dummies)
# Definiamo le fasce orarie
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay_Cat'] = df['HourDeparture'].apply(get_time_of_day)

# Creiamo le dummy variables (One-Hot Encoding)
time_dummies = pd.get_dummies(df['TimeOfDay_Cat'], prefix='TimeOfDay').astype(int)
df = pd.concat([df, time_dummies], axis=1)

# Assicuriamoci che ci siano tutte le colonne (anche se vuote) per coerenza
expected_times = ['TimeOfDay_Afternoon', 'TimeOfDay_Evening', 'TimeOfDay_Morning', 'TimeOfDay_Night']
for col in expected_times:
    if col not in df.columns:
        df[col] = 0

# ==============================================================================
# FASE 2: Calcolo dei Rate con LEAVE-ONE-OUT (Anti-Leakage)
# ==============================================================================

def calculate_loo_rate(dataframe, group_cols, target_col='Cancel', new_col_name='Rate'):
    """
    Calcola il tasso medio escludendo la riga corrente (Leave-One-Out).
    Formula: (Sum_Group - Target_Current) / (Count_Group - 1)
    """
    # Calcolo statistiche aggregate per gruppo
    # observed=True evita warning futuri con categorici
    agg = dataframe.groupby(group_cols, observed=True)[target_col].agg(['sum', 'count']).reset_index()
    
    # Unione temporanea
    merged = dataframe.merge(agg, on=group_cols, how='left')
    
    # Calcolo LOO
    # Se il conteggio è 1, il denominatore diventa 0 -> genererà NaN (corretto, poi riempiamo)
    numerator = merged['sum'] - merged[target_col]
    denominator = merged['count'] - 1
    
    # Calcolo
    loo_rates = numerator / denominator
    
    return loo_rates

# Calcoliamo la media globale per riempire i casi dove count=1 (divisione per zero)
global_mean = df['Cancel'].mean()

# 2.1 From_Rate
df['From_Rate'] = calculate_loo_rate(df, ['From'])
df['From_Rate'] = df['From_Rate'].fillna(global_mean)

# 2.2 To_Rate
df['To_Rate'] = calculate_loo_rate(df, ['To'])
df['To_Rate'] = df['To_Rate'].fillna(global_mean)

# 2.3 Route_Rate (Combinazione From + To)
df['Route_Rate'] = calculate_loo_rate(df, ['From', 'To'])
df['Route_Rate'] = df['Route_Rate'].fillna(global_mean)

# 2.4 User_Rate (Basato su NationalCode)
df['User_Rate'] = calculate_loo_rate(df, ['NationalCode'])
# Qui lo smoothing è cruciale: se un utente è nuovo o ha 1 sola prenotazione, usiamo la media globale
df['User_Rate'] = df['User_Rate'].fillna(global_mean)


# ==============================================================================
# FASE 3: Rate Complesso (Vehicle + Price Bin)
# ==============================================================================

# 3.1 Creazione Price Bins
df['PriceBin'] = pd.qcut(df['LogPrice'], q=5, labels=['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5'])

# 3.2 Identificazione Veicolo (serve una colonna unica per il groupby)
def get_vehicle_type(row):
    if row['Vehicle_Bus'] == 1: return 'Bus'
    if row['Vehicle_Plane'] == 1: return 'Plane'
    return 'Train' # Assumiamo Train se gli altri sono 0 o se Train è 1

df['temp_vehicle'] = df.apply(get_vehicle_type, axis=1)

# 3.3 Calcolo LOO
df['cancel_rate_per_vehicle_and_price'] = calculate_loo_rate(df, ['PriceBin', 'temp_vehicle'])
df['cancel_rate_per_vehicle_and_price'] = df['cancel_rate_per_vehicle_and_price'].fillna(global_mean)


# ==============================================================================
# FASE 4: Pulizia e Salvataggio Finale
# ==============================================================================

# Selezioniamo solo le colonne richieste nell'output finale
columns_to_keep = [
    'Domestic', 'TripReason', 'Cancel', 'LogLeadTime', 'LogPrice',
    'Vehicle_Bus', 'Vehicle_Plane', 'Vehicle_Train',
    'TimeOfDay_Afternoon', 'TimeOfDay_Evening', 'TimeOfDay_Morning', 'TimeOfDay_Night',
    'From_Rate', 'To_Rate', 'Route_Rate', 'User_Rate', 'cancel_rate_per_vehicle_and_price'
]

# Creiamo il dataset finale
df_final = df[columns_to_keep].copy()

# Aggiungiamo l'indice esplicito come prima colonna
df_final.reset_index(inplace=True)
df_final.rename(columns={'index': 'Index'}, inplace=True)

# Verifica rapida di Data Leakage (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_check = df_final.select_dtypes(include=[np.number]).drop(columns=['Index', 'Cancel'])
# Sostituiamo inf/nan per il check
X_check = X_check.replace([np.inf, -np.inf], 0).fillna(0)

print("\n--- Verifica Pre-Salvataggio ---")
print("Se vedi VIF > 1000, c'è ancora leakage. Se sono < 10-20, è perfetto.")
vif_sample = pd.DataFrame()
vif_sample["Variable"] = X_check.columns
try:
    vif_sample["VIF"] = [variance_inflation_factor(X_check.values, i) for i in range(X_check.shape[1])]
    print(vif_sample)
except Exception as e:
    print(f"Non ho potuto calcolare il VIF (forse costanti o errori numerici): {e}")

# Salvataggio
output_path = "./data/updated_data3.csv"
df_final.to_csv(output_path, index=False)

print(f"\nFile salvato correttamente in: {output_path}")
print(df_final.head())