
def smooth_mean(df, group_col, target_col, weight=20):
    
    global_cancel_mean = df[target_col].mean()
    
    # Calculate count and mean for each category
    stats = df.groupby(group_col)[target_col].agg(['count', 'mean'])
    
    # Apply smoothing formula
    counts = stats['count']
    means = stats['mean']
    smooth = (counts * means + weight * global_cancel_mean) / (counts + weight)
    
    return smooth, global_cancel_mean



def target_encoding(train_df, test_df, target_col = 'Cancel'):
    
    # Configuration: {Original_Column: (Variable_Name, Weight)}
    feature_config = {
        'From': ('From', 20),
        'To': ('To', 20),
        'Route': ('Route', 20),
        'NationalCode': ('User', 5)  # Maps NationalCode to "User"
    }

    # Create Route
    for df in [train_df, test_df]:
        df['Route'] = df['From'].astype(str) + ' to ' + df['To'].astype(str)

    # Feature engineering loop
    for col, (var_name, weight) in feature_config.items():
        # Calculate mapping ONLY on Training Set
        smooth_map, global_mean = smooth_mean(train_df, group_col=col, target_col=target_col, weight=weight)
        
        # Map the smoothed rate
        train_df[f'{var_name}_Rate'] = train_df[col].map(smooth_map)
        test_df[f'{var_name}_Rate'] = test_df[col].map(smooth_map)
        
        # Fill missing values
        test_df[f'{var_name}_Rate'] = test_df[f'{var_name}_Rate'].fillna(global_mean)
        
        # Fill missin values in train set just in case, but should not be necessary
        train_df[f'{var_name}_Rate'] = train_df[f'{var_name}_Rate'].fillna(global_mean)
        

    # Drop original columns
    cols_to_drop = list(feature_config.keys())
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)

    return test_df, train_df
