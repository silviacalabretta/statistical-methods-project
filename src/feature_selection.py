import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from data_utils import target_encoding, downsample_feature, create_time_of_day_feature, plot_feature_correction
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



# Standardize values
data['Vehicle'] = data['Vehicle'].replace('InternationalPlane', 'Plane')

# Date features
data['Created'] = pd.to_datetime(data['Created'])
data['DepartureTime'] = pd.to_datetime(data['DepartureTime'])

# Derived features
data['LogPrice']=np.log(data['Price'])

data['LeadTime_Days'] = (data['DepartureTime'] - data['Created']).dt.total_seconds() / 86400
data['LogLeadTime'] = np.log1p(data['LeadTime_Days'])
data['MonthDeparture'] = data['DepartureTime'].dt.month
data['HourDeparture'] = data['DepartureTime'].dt.hour

data = create_time_of_day_feature(data, hour_col='HourDeparture')

# Drop unused original columns
data=data.drop(columns=['DepartureTime','Created', 'LeadTime_Days', 'Price'])


# Binary Encoding
data['TripReason'] = data['TripReason'].map({'Work': 1, 'Int': 0})

# Label Encoding (Vehicle, TimeOfDay)
data['Vehicle'] = data['Vehicle'].map({'Bus': 0, 'Train': 1, 'Plane': 2})
data['TimeOfDay'] = data['TimeOfDay'].map({'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3})

# One-Hot Encoding
# cols_to_encode = ['Vehicle', 'TimeOfDay']

# encoder = OneHotEncoder(sparse_output=False, dtype=int)
# encoded_array = encoder.fit_transform(data[cols_to_encode])
# encoded_column = encoder.get_feature_names_out(cols_to_encode)

# encoded_data = pd.DataFrame(encoded_array, columns=encoded_column, index=data.index)

# data = data.drop(columns=cols_to_encode)
# data = pd.concat([data, encoded_data], axis=1)



train_df, test_df = train_test_split(
    data, 
    test_size=0.2, 
    random_state=19, 
    stratify=data['Cancel']
)


# Data correction: downsampling september cancelled tickets
# train_before = train_df.copy()     #uncomment for visualization

train_df = downsample_feature(
    df=train_df, 
    feature_col='MonthDeparture', 
    target_col='Cancel',
    category_value=9
)

# # Optional visualization
# plot_feature_correction(
#     df_original = train_before, 
#     df_corrected = train_df, 
#     feature_col='MonthDeparture', 
#     target_col='Cancel', 
#     conditional_col='Vehicle' 
# )

train_df=train_df.drop(columns=['MonthDeparture'])
test_df=test_df.drop(columns=['MonthDeparture'])


### Label Encoding (From, To, Route)

# Create a set of unique cities ONLY from Train
train_cities = set(train_df['From'].tolist() + train_df['To'].tolist())

# Create a manual mapping (City Name -> ID)
city_mapping = {city: idx for idx, city in enumerate(train_cities)}

# Apply to Train (Direct map)
train_df['From_Encoded'] = train_df['From'].map(city_mapping)
train_df['To_Encoded'] = train_df['To'].map(city_mapping)

# Apply to Test using .map() and fillna(-1)
# This handles unknown cities automatically by turning it into -1
test_df['From_Encoded'] = test_df['From'].map(city_mapping).fillna(-1).astype(int)
test_df['To_Encoded'] = test_df['To'].map(city_mapping).fillna(-1).astype(int)


# create and encode route
# data['Route'] = data['From'].astype(str) + ' to ' + data['To'].astype(str)
# data = encode_col(data, 'Route')

# Fit on train
train_df['Route'] = train_df['From'].astype(str) + ' to ' + train_df['To'].astype(str)
test_df['Route'] = test_df['From'].astype(str) + ' to ' + test_df['To'].astype(str)

# Create mapping from training data
route_mapping = {route: idx for idx, route in enumerate(train_df['Route'].unique())}

# Apply to train
train_df['Route_Encoded'] = train_df['Route'].map(route_mapping)

# Apply to test, filling unseen routes with -1
test_df['Route_Encoded'] = test_df['Route'].map(route_mapping).fillna(-1).astype(int)


#delete cols
train_df = train_df.drop(columns=['From','To', 'Route', 'NationalCode'])
test_df = test_df.drop(columns=['From','To', 'Route', 'NationalCode'])

print("\nDataset info:")
print(data.info())


# Target encoding
# train_df_encoded, test_df_encoded = target_encoding(train_df, test_df, target_col='Cancel')

# # Create the final arrays for the model
# X_train = train_df_encoded.drop(columns=['Cancel'])
# y_train = train_df_encoded['Cancel']

# X_test = test_df_encoded.drop(columns=['Cancel'])
# y_test = test_df_encoded['Cancel']



# Create the final arrays for the model
X_train = train_df.drop(columns=['Cancel'])
y_train = train_df['Cancel']

X_test = test_df.drop(columns=['Cancel'])
y_test = test_df['Cancel']



# Save datasets
output_dir = os.path.join(script_dir, '..', 'data')

os.makedirs(output_dir, exist_ok=True)

# data.to_csv(os.path.join(output_dir, 'updated_data.csv'), index=False)
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

print(f"\nCSV files saved successfully to {output_dir}")
print("\nX_train dataset info:")
print(X_train.info())