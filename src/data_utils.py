
def get_smoothed_rate(df, group_col, target_col, weight, global_mean):
    
    # Calculate count and mean for each category
    stats = df.groupby(group_col)[target_col].agg(['count', 'mean'])
    
    # Apply smoothing formula
    counts = stats['count']
    means = stats['mean']
    smooth = (counts * means + weight * global_mean) / (counts + weight)
    
    return smooth

def get_loo_smoothed_rate(df, group_col, target_col, weight, global_mean):
    """
    Calculates the Smoothed Rate using Leave-One-Out (LOO) logic.
    Returns a pandas Series aligned with the original DataFrame index.
    """
    # Compute total sum and count for the group (aligned to rows)
    total_sum = df.groupby(group_col)[target_col].transform('sum')
    total_count = df.groupby(group_col)[target_col].transform('count')
    
    # Leave-One-Out: subtract the current row's information
    loo_sum = total_sum - df[target_col]
    loo_count = total_count - 1
    
    # Apply the smoothing formula
    smoothed_rate = (loo_sum + weight * global_mean) / (loo_count + weight)
    
    return smoothed_rate

def target_encoding(train_df, test_df, target_col = 'Cancel'):
    """
    Applies Target Encoding:
    - Train Set: Uses Leave-One-Out (LOO) to prevent data leakage.
    - Test Set: Uses standard mapping derived from the full Train Set.
    """
    
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Configuration: {Original_Column: (Variable_Name, Weight)}
    feature_config = {
        'From': ('From', 20),
        'To': ('To', 20),
        'Route': ('Route', 20),
        'NationalCode': ('User', 5)  # Maps NationalCode to "User"
    }

    # Calculate global mean from train only
    global_mean = train_df[target_col].mean()
    
    # Create Route
    for df in [train_df, test_df]:
        df['Route'] = df['From'].astype(str) + ' to ' + df['To'].astype(str)

    # Encoding loop
    for col, (var_name, weight) in feature_config.items():

        # Encoding Training Set
        train_df[f'{var_name}_Rate'] = get_loo_smoothed_rate(
            train_df, 
            group_col=col, 
            target_col=target_col, 
            weight=weight, 
            global_mean=global_mean
        )

        # Fill NaNs in train set (if an element has only 1 occurrence, loo_count is 0)
        train_df[f'{var_name}_Rate'] = train_df[f'{var_name}_Rate'].fillna(global_mean)
        

        # Encoding Test set
        smooth_map = get_smoothed_rate(
            train_df, 
            group_col=col, 
            target_col=target_col, 
            weight=weight,
            global_mean=global_mean
        )
        
        test_df[f'{var_name}_Rate'] = test_df[col].map(smooth_map)
        
        # Fill missing values
        test_df[f'{var_name}_Rate'] = test_df[f'{var_name}_Rate'].fillna(global_mean)
            

    # Drop original columns
    cols_to_drop = list(feature_config.keys())
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)

    return test_df, train_df