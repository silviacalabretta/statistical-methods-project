import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

data = pd.read_csv("./data/updated_data.csv")
data.columns = data.columns.str.strip()

# choose between "To" r "From"
FromOrTo = "From" 
name=""
if(FromOrTo=="From"):name="Departed" 
else: name="Arrived"

# 2. Pre-calcolo delle maschere booleane per i veicoli
is_train = data['VehicleClass'] == 2
is_plane = (data['VehicleClass'] != 2) & (data['Vehicle_Plane'] == True)
is_int_plane = (data['VehicleClass'] != 2) & (data['Vehicle_IntPlane'] == True)
is_bus = (data['VehicleClass'] != 2) & (data['Vehicle_Plane'] == False) & (data['Vehicle_IntPlane'] == False)

# 3. creating summary table for departure or returning
byCity = data.groupby(FromOrTo).agg(
    CancellationRate=('Cancel', 'mean'),
    NumberOfCancellation=('Cancel', 'sum'),
    DomesticDepRate=('Domestic', 'mean'),
    meanCost=('Price', 'mean'),
    TotalDepartures=(FromOrTo, 'count')
)

# 4. added countigs per vehicle types
nametrain=str.join(['numberOfTrain',name])
nameplane=str.join(['numberOfPlanes',name])
nameintplanes=str.join(['numberOfIntPlanes',name])
namebus=str.join(['numberOfBus',name])
byCity[nametrain] = data[is_train].groupby(FromOrTo)[FromOrTo].count()
byCity[nameplane] = data[is_plane].groupby(FromOrTo)[FromOrTo].count()
byCity[nameintplanes] = data[is_int_plane].groupby(FromOrTo)[FromOrTo].count()
byCity[namebus] = data[is_bus].groupby(FromOrTo)[FromOrTo].count()

# 5. final cleaning
byCity = byCity.fillna(0).reset_index()

# --- minimum of 20
min_trips_threshold = 20 
byCity_filtered = byCity[byCity['TotalDepartures'] >= min_trips_threshold].copy()

print(f"analysis of {name} - cities counted: {len(byCity_filtered)} (excluded {len(byCity) - len(byCity_filtered)} under the minimum thrashold)")

# 6. graphing
numeric_cols = byCity_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'CancellationRate' in numeric_cols:
    numeric_cols.remove('CancellationRate')
if FromOrTo in numeric_cols:
    numeric_cols.remove(FromOrTo)

n_cols = 3
n_rows = math.ceil(len(numeric_cols) / n_cols)

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 6 * n_rows))
fig.suptitle(f"analysis in {name} (minimum {min_trips_threshold} trips)", fontsize=20, y=1.02)

axes = axes.flatten()

# list of column for log scale 
cols_to_log = ['TotalDepartures', nametrain, nameplane, 
               nameintplanes, namebus, 'NumberOfCancellation']

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.scatterplot(data=byCity_filtered, x=col, y='CancellationRate', alpha=0.6, ax=ax)
    
    if col in cols_to_log:
        ax.set_xscale('log')
        ax.set_xlabel(f"{col} log scale")
    else:
        ax.set_xlabel(col)

    ax.set_title(f"Rate vs {col}")
    ax.set_ylabel("Cancellation Rate")
    ax.grid(True, which="both", ls="-", alpha=0.2)

# hide empty
for k in range(i + 1, len(axes)):
    axes[k].axis('off')

plt.tight_layout()
plt.savefig("filtered_departure_analysis_of_cancellations.png", bbox_inches='tight')
plt.show()