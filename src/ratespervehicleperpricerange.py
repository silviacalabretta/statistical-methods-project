import pandas as pd
from tabulate import tabulate

# Load the dataset
df = pd.read_csv("./data/updated_data.csv")

# Create 'PriceBin' using pd.qcut
df['PriceBin'] = pd.qcut(df['LogPrice'], q=5, labels=['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5'])

# Group by 'PriceBin' and sum the tickets for each vehicle type
vehicle_tickets_by_bin = df.groupby('PriceBin', observed=True).agg(
    Total_Tickets=('Cancel', 'count'),
    Bus_Tickets=('Vehicle_Bus', 'sum'),
    Plane_Tickets=('Vehicle_Plane', 'sum'),
    Train_Tickets=('Vehicle_Train', 'sum'),
    Cancelled_total=('Cancel', 'sum'),
    Cancelled_Bus=('Vehicle_Bus', lambda x: (x * df.loc[x.index, 'Cancel']).sum()),
    Cancelled_Plane=('Vehicle_Plane', lambda x: (x * df.loc[x.index, 'Cancel']).sum()),
    Cancelled_Train=('Vehicle_Train', lambda x: (x * df.loc[x.index, 'Cancel']).sum())
).reset_index()

# Calculate cancellation rate for each vehicle type within each bin
vehicle_tickets_by_bin["Cancel_rate"] = vehicle_tickets_by_bin["Cancelled_total"] / vehicle_tickets_by_bin["Total_Tickets"]
vehicle_tickets_by_bin['Cancel_Rate_Bus'] = vehicle_tickets_by_bin['Cancelled_Bus'] / vehicle_tickets_by_bin['Bus_Tickets']
vehicle_tickets_by_bin['Cancel_Rate_Plane'] = vehicle_tickets_by_bin['Cancelled_Plane'] / vehicle_tickets_by_bin['Plane_Tickets']
vehicle_tickets_by_bin['Cancel_Rate_Train'] = vehicle_tickets_by_bin['Cancelled_Train'] / vehicle_tickets_by_bin['Train_Tickets']

# Assign the cancellation rate based on the vehicle type and price bin
df['cancel_rate_per_vehicle_and_price'] = df.apply(
    lambda row: vehicle_tickets_by_bin.loc[vehicle_tickets_by_bin['PriceBin'] == row['PriceBin'], 'Cancel_Rate_Bus'].values[0]
    if row['Vehicle_Bus'] == 1
    else vehicle_tickets_by_bin.loc[vehicle_tickets_by_bin['PriceBin'] == row['PriceBin'], 'Cancel_Rate_Plane'].values[0]
    if row['Vehicle_Plane'] == 1
    else vehicle_tickets_by_bin.loc[vehicle_tickets_by_bin['PriceBin'] == row['PriceBin'], 'Cancel_Rate_Train'].values[0],
    axis=1
)
print(tabulate(df[:50],df.columns))
print(tabulate(vehicle_tickets_by_bin,vehicle_tickets_by_bin.columns))

df=df.drop("PriceBin",axis=1)
df.to_csv("./data/updated_data2.csv")

predictors = df.drop(columns=["Cancel"])
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = predictors.columns
vif_data["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

print(vif_data)