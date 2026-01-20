import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you already have your data loaded in 'df'
df = pd.read_csv('../data/tickets_data.csv')

# Your actual column names based on the screenshot
# Extract month from DepartureTime
df['Month'] = pd.to_datetime(df['DepartureTime']).dt.month

# Create cancellation indicator (1 if cancelled, 0 if not)
df['Cancelled'] = df['CancelTime'].notna().astype(int)

# Column names
month_col = 'Month'
cancel_col = 'Cancelled'
vehicle_col = 'Vehicle'  # Adjust if your vehicle column has a different name

# Set random seed for reproducibility
np.random.seed(42)

# Get September data
september_data = df[df[month_col] == 9]
original_cancel_rate = september_data[cancel_col].mean()
original_cancelled = september_data[cancel_col].sum()

# Calculate target cancellation rate (average of other months)
target_rate = df[df[month_col] != 9][cancel_col].mean()

# Calculate how many cancelled tickets to delete
target_cancelled = int(target_rate * len(september_data))
num_to_delete = int(original_cancelled) - target_cancelled

print(f"September - Before: {int(original_cancelled)} cancelled out of {len(september_data)} ({original_cancel_rate:.2%})")
print(f"Target rate: {target_rate:.2%}")
print(f"Need to delete: {num_to_delete} cancelled tickets from September")

# Create copy and delete random cancelled tickets from September
df_modified = df.copy()

if num_to_delete > 0:
    # Get indices of September cancelled tickets
    sept_cancelled_indices = df_modified[
        (df_modified[month_col] == 9) & 
        (df_modified[cancel_col] == 1)
    ].index.tolist()
    
    # Randomly select and delete
    indices_to_delete = np.random.choice(
        sept_cancelled_indices, 
        size=num_to_delete, 
        replace=False
    )
    df_modified = df_modified.drop(indices_to_delete).reset_index(drop=True)
    
    # Verify results
    september_new = df_modified[df_modified[month_col] == 9]
    new_cancel_rate = september_new[cancel_col].mean()
    new_cancelled = int(september_new[cancel_col].sum())
    
    print(f"\nSeptember - After: {new_cancelled} cancelled out of {len(september_new)} ({new_cancel_rate:.2%})")
    print(f"Deleted {len(df) - len(df_modified)} rows total")
    print(f"Cancellation rate reduced by: {(original_cancel_rate - new_cancel_rate)*100:.2f} percentage points")

# ========================================
# CREATE STACKED BARPLOTS (1x2) - ONE BAR PER MONTH
# Each vehicle's contribution to the total cancellation rate
# ========================================

# Get unique vehicles and months
vehicles = sorted(df[vehicle_col].unique())
months = sorted(df[month_col].unique())

# Set up colors for each vehicle type
colors_map = {'Bus': '#e74c3c', 'Train': '#3498db', 'Plane': '#2ecc71', 'IntPlane': '#f39c12'}
vehicle_colors = {vehicle: colors_map.get(vehicle, '#95a5a6') for vehicle in vehicles}

# Function to prepare data for stacked bars
# Each vehicle's proportion = (cancelled_vehicle / total_tickets_month)
def prepare_stacked_data(data):
    result = {}
    for month in months:
        result[month] = {}
        month_data = data[data[month_col] == month]
        total_tickets = len(month_data)
        
        if total_tickets > 0:
            for vehicle in vehicles:
                vehicle_data = month_data[month_data[vehicle_col] == vehicle]
                cancelled_vehicle = vehicle_data[cancel_col].sum()
                # Contribution of this vehicle to total cancellation rate
                result[month][vehicle] = cancelled_vehicle / total_tickets
        else:
            for vehicle in vehicles:
                result[month][vehicle] = 0
    
    return pd.DataFrame(result).T

# Prepare data for BEFORE and AFTER
data_before = prepare_stacked_data(df)
data_after = prepare_stacked_data(df_modified)

# Create figure with 1x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Function to create stacked bar plot (ONE bar per month, colored by vehicle)
def plot_stacked_bars(ax, data, title):
    x = np.arange(len(data.index))
    width = 0.6

    bottom = np.zeros(len(data.index))

    for vehicle in vehicles:
        if vehicle in data.columns:
            values = data[vehicle].values
            bars = ax.bar(
                x,
                values,
                width,
                label=vehicle,                 # labels for legend
                bottom=bottom,
                color=vehicle_colors[vehicle],
                alpha=0.9,
                edgecolor='white',
                linewidth=2,
            )
            bottom += values

    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancellation Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data.index)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

    # Calculate overall cancellation rate for each month
    total_rates = bottom
    ax.set_ylim(0, total_rates.max() * 1.15)

    # Add text showing total rate on top of each bar
    for i, (idx, total) in enumerate(zip(data.index, total_rates)):
        ax.text(
            i,
            total + 0.01,
            f'{total:.2%}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

# Plot BEFORE and AFTER
plot_stacked_bars(axes[0], data_before, 'Cancellation Rate by Month and Vehicle (BEFORE)')
plot_stacked_bars(axes[1], data_after, 'Cancellation Rate by Month and Vehicle (AFTER)')

# ===== NEW PART: legend outside the plots =====
# Get legend handles/labels from one axis
handles, labels = axes[0].get_legend_handles_labels()

# Put ONE shared legend to the right of both subplots
fig.legend(
    handles,
    labels,
    title='Vehicle Type',
    loc='upper left',
    bbox_to_anchor=(0.84, 0.93),   # (x, y) outside the figure on the right
    framealpha=0.95,
)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for the legend
plt.savefig('cancellation_comparison_stacked_by_month.png', dpi=300, bbox_inches='tight')
plt.show()
