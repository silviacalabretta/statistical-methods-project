import pandas as pd
# from sklearn.model_selection import train_test_split

def create_user_history(
        train_df, 
        test_df, 
        target_col='Cancel'
    ):

    id_col='NationalCode'

    # Calculate user statistics (only on training data)
    user_stats = train_df.groupby(id_col)[target_col].agg(['mean', 'count']).reset_index()
    user_stats.columns = [id_col, 'User_Cancel_Rate', 'User_Total_Tickets']
    global_cancel_mean = train_df[target_col].mean()
    print(f"Global Training Cancel Rate: {global_cancel_mean:.4f}")
    
    # Merge stats into Train and Test
    train_df = train_df.merge(user_stats, on=id_col, how='left')
    test_df = test_df.merge(user_stats, on=id_col, how='left')
    
    # Users in Test who never appeared in Train get 0 history
    new_users_count = test_df['User_Total_Tickets'].isna().sum()
    print(f"Found {new_users_count} new users in Test set")
    
    # Cancel rate = average cancel rate in train set (bayesian prior)
    test_df['User_Cancel_Rate'] = test_df['User_Cancel_Rate'].fillna(global_cancel_mean)
    # Total tickets = 0
    test_df['User_Total_Tickets'] = test_df['User_Total_Tickets'].fillna(0).astype(int)

    train_df = train_df.drop(columns=[id_col])
    test_df = test_df.drop(columns=[id_col])

    return train_df, test_df


def calc_smooth_mean(df, group_col, target_col, weight=20):
    
    global_cancel_mean = df[target_col].mean()
    
    # Calculate count and mean for each category
    stats = df.groupby(group_col)[target_col].agg(['count', 'mean'])
    
    # Apply smoothing formula
    counts = stats['count']
    means = stats['mean']
    smooth = (counts * means + weight * global_cancel_mean) / (counts + weight)
    
    return smooth, global_cancel_mean


def create_routes(
        train_df, 
        test_df, 
        target_col='Cancel',
        weight = 20
    ):

    # Create the route
    train_df['Route'] = train_df['From'].astype(str) + ' to ' + train_df['To'].astype(str)
    test_df['Route'] = test_df['From'].astype(str) + ' to ' + test_df['To'].astype(str)


    # Compute the cancellation rate in 'From', 'To', and 'Route'
    encoding_maps = {}
    global_means = {}

    for col in ['From', 'To', 'Route']:
        # Calculate mapping on TRAIN set
        smooth_map, glob_mean = calc_smooth_mean(train_df, group_col=col, target_col=target_col, weight=weight)
        
        # Store them for later use
        encoding_maps[col] = smooth_map
        global_means[col] = glob_mean
        
        # Map to X_train
        train_df[f'{col}_Rate'] = train_df[col].map(smooth_map)
        
        # Fill missing in train (should not be required)
        train_df[f'{col}_Rate'] = train_df[f'{col}_Rate'].fillna(glob_mean)

        # 4. Map to test_df 
        test_df[f'{col}_Rate'] = test_df[col].map(smooth_map)
        
        # If a route appears in Test but NOT in Train, fill with Train Global Mean
        test_df[f'{col}_Rate'] = test_df[f'{col}_Rate'].fillna(glob_mean)

    # train_df and test_df are ready with numerical columns
    print(train_df[['Route', 'Route_Rate']].head())

    return train_df, test_df


def time_based_train_test_split(
        df, 
        date_col='Created',
        target_col='Cancel',
        
        train_size=0.8
    ):
    """
    Splits data by time, calculates user history on TRAIN only, 
    and handles new users in TEST by setting history to 0.
    """
    

    print(f"Sorting data by {date_col}...")
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Split the data
    split_index = int(len(df) * train_size)
    train_df = df.iloc[:split_index].copy()
    test_df  = df.iloc[split_index:].copy()
    
    print(f"Split complete. Training: {len(train_df)} rows, Test: {len(test_df)} rows.")
    
    # Clean-up: drop the Date column
    train_df = train_df.drop(columns=[date_col])
    test_df = test_df.drop(columns=[date_col])

    train_df, test_df = create_user_history(train_df, test_df, target_col=target_col)
    train_df, test_df = create_routes(train_df, test_df, target_col=target_col)
    
    return train_df, test_df