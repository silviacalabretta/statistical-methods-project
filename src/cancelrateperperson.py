
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

# calculating number of cancellations per vehicle
cancel_by_vehicle = df[df['Cancel'] == 1].groupby('NationalCode').agg(
    Cancelled_Bus=('Vehicle_Bus', 'sum'),
    Cancelled_Plane=('Vehicle_Plane', 'sum'),
    Cancelled_Train=('Vehicle_Train', 'sum'),
).reset_index()

# uniting info
final_summary = pd.merge(summary, cancel_by_vehicle, on='NationalCode', how='left').fillna(0)
#avoiding some outliers
reduced = final_summary[final_summary["Total_Tickets"] > 5]

# Creiamo il grafico a dispersione
fig, ax = plt.subplots(figsize=(10, 6))


scatter = ax.scatter(
    reduced["Total_Tickets"], 
    reduced["Cancel_Rate"], 
    c='darkblue',  
    s=100,         
    alpha=0.7,     
    edgecolors='black'
)
from tabulate import tabulate
a=final_summary[final_summary['Cancel_Rate']==1]
print(tabulate(a[a["Total_Tickets"]>4],final_summary.columns))
print(len(a[a["Total_Tickets"]>4]))

ax.set_title('Relationship between number of tickets and cancellation rate', fontsize=14, pad=20)
ax.set_xlabel('Total number of tickets', fontsize=12)
ax.set_ylabel('cancellation rate', fontsize=12)
plt.tight_layout()
plt.show()