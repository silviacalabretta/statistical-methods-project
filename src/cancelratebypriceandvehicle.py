import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Load the dataset
df = pd.read_csv('./data/updated_data.csv')
df=df[df['MonthDeparture'] != 9]
# --- 1. PREPARE THE BINS ---
# Based on your bins [0, 11, 12...], this matches the 'LogPrice' column.
# (Raw 'Price' is in millions, so we use LogPrice for these small bin numbers)
bins = [0, 12, 13, 14, 15, 16, 17, float('inf')]
labels = ['0-12', '12-13', '13-14', '14-15', '15-16', '16-17', '17+']

df['Price_Range'] = pd.cut(df['LogPrice'], bins=bins, labels=labels, right=False)

# --- 2. CALCULATE PERCENTAGES ---
# We want: (Cancelled Tickets of Vehicle X / TOTAL Tickets in Range) * 100

# First, get the Total count of tickets (cancelled or not) for each price range
range_totals = df.groupby('Price_Range', observed=False)['Cancel'].count().rename('Total_Tickets_In_Range')

# Second, get the count of CANCELLED tickets for each vehicle in each range
vehicle_cancels = df[df['Cancel'] == 1].groupby(['Price_Range', 'Vehicle'], observed=False)['Cancel'].count()

# Unstack to get vehicles as columns (0, 1, 2, 3) and fill missing with 0
df_plot = vehicle_cancels.unstack(level='Vehicle').fillna(0)

# Ensure all vehicle columns exist (0, 1, 2, 3)
for v in [0, 1, 2, 3]:
    if v not in df_plot.columns:
        df_plot[v] = 0

# Divide by the range totals to get the contribution percentage
df_plot = df_plot.div(range_totals, axis=0) * 100

# --- 3. PLOTTING ---
plt.figure(figsize=(12, 8))

# Define colors and labels map
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
vehicle_labels = {0: 'Bus', 1: 'Train', 2: 'Plane', 3: 'Int Plane'}

# Create the stacked bar chart
# We iterate to plot manually or use pandas plot. Pandas plot is cleaner here.
ax = df_plot.plot(kind='bar', stacked=True, color=colors, figsize=(12, 7), width=0.7)

# Title and Labels
plt.title('Cancellation Rate by Price Range without september\n(Breakdown by Vehicle Contribution)', fontsize=14)
plt.xlabel('Price Range (Log Scale)', fontsize=12)
plt.ylabel('Percentage of Tickets Cancelled (%)', fontsize=12)
plt.xticks(rotation=0)

# --- 4. LEGEND & ANNOTATIONS ---
# Custom Legend
handles, _ = ax.get_legend_handles_labels()
plt.legend(handles, [vehicle_labels[int(h.get_label())] for h in handles], title='Vehicle Type')

# Add percentage labels inside the bars
# for c in ax.containers:
#     # Only label if the segment is large enough to fit text (e.g., > 0.5%)
#     labels = [f'{v.get_height():.1f}%' if v.get_height() > 0.5 else '' for v in c]
#     ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=9, fontweight='bold')


cancel_stats = df.groupby('Price_Range', observed=False)['Cancel'].agg(
    cancel_rate='mean',
    total_cancellations='sum',
    total_tickets='count'
).reset_index()

for i, (idx, row) in enumerate(cancel_stats.iterrows()):
    ax.text(i, 0.01, f"n={int(row['total_tickets']):,} \n c={int(row['total_cancellations']):,}", 
            ha='center', va='bottom', fontsize=10, weight='bold', 
            color='white', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
# 4. MERGED LEGEND
# Get standard vehicle handles
handles, _ = ax.get_legend_handles_labels()
v_names = [vehicle_labels[int(h.get_label())] for h in handles]

# Create custom patches for c and n definitions
info_patches = [
    mpatches.Patch(color='none', label='c = number of cancelled tickets'),
    mpatches.Patch(color='none', label='n = total tickets sold in range')
]

# Combine all handles and labels into one list
all_handles = handles + info_patches
all_labels = v_names + [p.get_label() for p in info_patches]

# Apply the merged legend
ax.legend(handles=all_handles, labels=all_labels, title='Vehicle Type & Key', 
          loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
# Add Total labels on top of the bars
totals = df_plot.sum(axis=1)
for i, total in enumerate(totals):
    if total > 0:
        ax.text(i, total + 0.2, f'{total:.1f}%', ha='center', va='bottom', fontweight='bold', color='black')

plt.tight_layout()
plt.show()