
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    return train_df, test_df


def downsample_feature(df, feature_col='MonthDeparture', target_col='Cancel', 
                                      category_value=9, random_seed=42):
    """
    Reduces the cancellation rate of a specific category to match the average 
    rate of the rest of the year by randomly dropping cancelled tickets.
    """

    # Create a copy to avoid modifying the original dataframe inplace
    df_mod = df.copy()
    
    # Get stats for the specific category
    category_mask = df_mod[feature_col] == category_value
    category_data = df_mod[category_mask]
    original_target_rate = category_data[target_col].mean()
    
    if len(category_data) == 0:
        print(f"No data found for {feature_col} category {category_value}. Returning original dataframe.")
        return df_mod

    # Current positive count in category_value (number of rows with 'Cancel'=1)
    original_target_count = category_data[target_col].sum()
    
    # Calculate goal rate (average of all OTHER categories)
    other_categories_mask = df_mod[feature_col] != category_value
    goal_rate = df_mod[other_categories_mask][target_col].mean()
    
    # Calculate approximately how many to delete: 
    # x = original_target_count - goal_rate * len(category_data)
    # Final count of positive = Total * Goal Rate

    # Cancellation rate the category value should have 
    goal_target_count = int(goal_rate * len(category_data))

    # Number of rows to delete to reach the target cancellation rate
    num_to_delete = int(original_target_count) - goal_target_count
    
    print(f"--- Correction for {feature_col} category {category_value} ---")
    print(f"Target Rate: {goal_rate:.2%}")
    print(f"Current Rate: {category_data[target_col].mean():.2%}")
    
    if num_to_delete > 0:
        print(f"Removing {num_to_delete} cancelled tickets to align rates...")
        
        # Set seed
        np.random.seed(random_seed)
        
        # Identify candidates to drop (category == target AND Cancel == 1)
        candidates = df_mod[category_mask & (df_mod[target_col] == 1)].index.tolist()
        
        # Random sample
        indices_to_delete = np.random.choice(candidates, size=num_to_delete, replace=False)
        
        # Drop rows
        df_mod = df_mod.drop(indices_to_delete).reset_index(drop=True)
        
        # Validation
        new_rate = df_mod[df_mod[feature_col] == category_value][target_col].mean()
        print(f"New Rate: {new_rate:.2%}")
        print(f"Cancellation rate reduced by: {(original_target_rate - new_rate)*100:.2f} percentage points")
    else:
        print("No deletion needed (Current rate is already lower or equal to target).")

    return df_mod


def plot_feature_correction(df_original, df_corrected, 
                             feature_col='MonthDeparture', 
                             target_col='Cancel', 
                             conditional_col='Vehicle'):
    """
    Plots a side-by-side comparison of cancellation rates per month/vehicle 
    before and after the correction.
    """
    
    # --- Helper to prepare data ---
    def prepare_data(data):
        months = sorted(data[feature_col].unique())
        vehicles = sorted(data[conditional_col].unique())
        result = {}
        
        for m in months:
            result[m] = {}
            m_data = data[data[feature_col] == m]
            total = len(m_data)
            if total > 0:
                for v in vehicles:
                    v_cancelled = m_data[(m_data[conditional_col] == v) & (m_data[target_col] == 1)].shape[0]
                    result[m][v] = v_cancelled / total
            else:
                for v in vehicles:
                    result[m][v] = 0
        return pd.DataFrame(result).T, vehicles

    # Prepare Data
    data_before, vehicles = prepare_data(df_original)
    data_after, _ = prepare_data(df_corrected)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors (Bus/Train/Plane specific, fallback to generic if names differ)
    color_map = {'Bus': '#e74c3c', 'Train': '#3498db', 'Plane': '#2ecc71', 'IntPlane': '#f39c12'}
    # Fallback generator for unknown vehicle types
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(vehicles)))
    
    def plot_ax(ax, data, title):
        x = np.arange(len(data.index))
        bottom = np.zeros(len(data.index))
        
        for i, v in enumerate(vehicles):
            if v in data.columns:
                values = data[v].values
                color = color_map.get(v, default_colors[i])
                ax.bar(x, values, 0.6, bottom=bottom, label=v, color=color, alpha=0.9, edgecolor='white')
                bottom += values
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Cancellation Rate')
        ax.set_xticks(x)
        ax.set_xticklabels(data.index)
        ax.set_ylim(0, bottom.max() * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add totals labels
        for i, total in enumerate(bottom):
            ax.text(i, total + 0.005, f'{total:.1%}', ha='center', va='bottom', fontsize=9)

    plot_ax(axes[0], data_before, 'Before Correction')
    plot_ax(axes[1], data_after, 'After Correction')
    
    # Single Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Vehicle', loc='upper right', bbox_to_anchor=(0.98, 0.9))
    
    plt.tight_layout()
    plt.show()


def create_time_of_day_feature(df, hour_col='HourDeparture'):
    """
    Transforms numerical hour (0-23) into 4 categorical bins:
    Morning, Afternoon, Evening, Night.
    """
    df = df.copy()
    
    # Logic: 
    # Morning: 06:00 - 11:59
    # Afternoon: 12:00 - 17:59
    # Evening: 18:00 - 22:59
    # Night: 23:00 - 05:59 (Handles the wrap-around)
    
    conditions = [
        (df[hour_col].between(6, 11)),
        (df[hour_col].between(12, 17)),
        (df[hour_col].between(18, 22))
    ]
    choices = ['Morning', 'Afternoon', 'Evening']
    
    # Vectorized selection; default is 'Night' (covers 23, 0, 1, 2, 3, 4, 5)
    df['TimeOfDay'] = np.select(conditions, choices, default='Night')
    
    # Drop original hour column to avoid collinearity
    df = df.drop(columns=[hour_col])
    
    return df