import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from data_utils import target_encoding
from city_translation import city_map



script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, '..', 'data', 'tickets_data.csv')
data = pd.read_csv(file_path)

# City names translation 
data['From'] = data['From'].map(city_map).fillna(data['From'])
data['To'] = data['To'].map(city_map).fillna(data['To'])

# Data Cleaning
data = data[data['Price'] > 0]
data = data[data['CouponDiscount'] >= 0]
data = data[data['CouponDiscount'] <= data['Price']]

# drop useless columns
data = data.drop(columns=[
    'HashPassportNumber_p', 
    'UserID', 
    'HashEmail', 
    'BillID', 
    'BuyerMobile', 
    'TicketID', 
    'CancelTime', 
    'ReserveStatus', 
    'VehicleClass', 
    'Male', 
    'CouponDiscount', 
    'VehicleType'])



# Encoding
data['TripReason'] = data['TripReason'].map({'Work': 1, 'Int': 0})
data['Vehicle'] = data['Vehicle'].replace('InternationalPlane', 'Plane')

encoder = OneHotEncoder(sparse_output=False, dtype=int)
encoded_array = encoder.fit_transform(data[['Vehicle']])

encoded_column = encoder.get_feature_names_out(['Vehicle'])
encoded_data = pd.DataFrame(encoded_array, columns=encoded_column)
encoded_data.index = data.index
data_encoded = data.drop(columns=['Vehicle'])
data = pd.concat([data_encoded, encoded_data], axis=1)

# Feature Engineering
data['LogPrice']=np.log(data['Price'])

data['Created'] = pd.to_datetime(data['Created'])
data['DepartureTime'] = pd.to_datetime(data['DepartureTime'])
data['LeadTime_Days'] = (data['DepartureTime'] - data['Created']).dt.total_seconds() / 86400
data['MonthDeparture'] = data['DepartureTime'].dt.month
data['HourDeparture'] = data['DepartureTime'].dt.hour

data=data.drop(columns=['DepartureTime','Created', 'Price'])

### ADD THE SEPTEMBER CLEANING


# Split the whole dataframe. 'Cancel' is still inside train_df and test_df
train_df, test_df = train_test_split(
    data, 
    test_size=0.2, 
    random_state=19, 
    stratify=data['Cancel']
)

# Target encoding
train_df_encoded, test_df_encoded = target_encoding(train_df, test_df, target_col='Cancel')

# Create the final arrays for the model
X_train = train_df_encoded.drop(columns=['Cancel'])
y_train = train_df_encoded['Cancel']

X_test = test_df_encoded.drop(columns=['Cancel'])
y_test = test_df_encoded['Cancel']


# Save datasets
output_dir = os.path.join(script_dir, '..', 'data')

os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

print(f"CSV files saved successfully to {output_dir}")
print(X_train.info())