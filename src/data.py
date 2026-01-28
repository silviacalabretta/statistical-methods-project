def time_based_train_test_split(
        df, 
        id_col=0, 
        target_col='Cancel', 
        train_size=0.8
    ):
    """
    Splits data by time, calculates user history on TRAIN only, 
    and handles new users in TEST by setting history to 0.
    """

    # Split the data
    split_index = int(len(df) * train_size)
    train_df = df.iloc[:split_index].copy()
    test_df  = df.iloc[split_index:].copy()
    
    print(f"Split complete. Training: {len(train_df)} rows, Test: {len(test_df)} rows.")
    
    # Calculate user statistics (only on training data)
    user_stats = train_df.groupby(id_col)[target_col].agg(['mean', 'count']).reset_index()
    user_stats.columns = [id_col, 'User_Cancel_Rate', 'User_Total_Tickets']
    
    # Merge stats into Train and Test
    train_df = train_df.merge(user_stats, on=id_col, how='left')
    test_df = test_df.merge(user_stats, on=id_col, how='left')
    
    # Users in Test who never appeared in Train get 0 history
    
    new_users_count = test_df['User_Total_Tickets'].isna().sum()
    print(f"Found {new_users_count} new users in Test set (setting their history to 0).")
    
    test_df['User_Cancel_Rate'] = test_df['User_Cancel_Rate'].fillna(0)
    test_df['User_Total_Tickets'] = test_df['User_Total_Tickets'].fillna(0).astype(int)
    
    # Final Clean-up: drop the ID and the Date column
    train_df = train_df.drop(columns=[id_col])
    test_df = test_df.drop(columns=[id_col])
    
    return train_df, test_df