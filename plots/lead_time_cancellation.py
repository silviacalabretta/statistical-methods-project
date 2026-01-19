#===========================================
# LEAD TIME ANALYSIS - CANCELLATION PATTERNS
# (FIXED VERSION - Clean X-axis labels)
#===========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./data/updated_data.csv')  # Load your dataset
save_dir = 'plots/'  # Directory to save plots


# Set style
sns.set_style("whitegrid")

# Create Lead Time Categories with SINGLE-LINE labels (no \n)
data['LeadTime_Category'] = pd.cut(data['LeadTime_Days'], 
                                           bins=[-1, 1, 7, 30, 100],
                                           labels=['Last-Minute (<1 day)', 
                                                  'Short (1-7 days)', 
                                                  'Medium (7-30 days)', 
                                                  'Long (>30 days)'])

# Calculate statistics
leadtime_cancel = data.groupby('LeadTime_Category', observed=True)['Cancel'].agg(['mean', 'count', 'sum'])
leadtime_cancel.columns = ['Cancel_Rate', 'Total_Tickets', 'Cancelled']

# Create single plot with more vertical space
fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax.bar(range(len(leadtime_cancel)), leadtime_cancel['Cancel_Rate'], 
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)

ax.set_title('Cancellation Rate by Lead Time Category', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Lead Time Category', fontsize=13, weight='bold', labelpad=15)
ax.set_ylabel('Cancellation Rate', fontsize=13, weight='bold')
ax.set_xticks(range(len(leadtime_cancel)))

# Set x-axis labels WITHOUT rotation
ax.set_xticklabels(leadtime_cancel.index, fontsize=11, ha='center')

ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add overall mean line
overall_mean = data['Cancel'].mean()
ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
           label=f'Overall Mean ({overall_mean:.3f})')

# Add percentage labels on TOP of bars
for i, (idx, row) in enumerate(leadtime_cancel.iterrows()):
    ax.text(i, row['Cancel_Rate'] + 0.008, f"{row['Cancel_Rate']:.1%}", 
            ha='center', va='bottom', fontsize=13, weight='bold', color='black')

# Add sample size INSIDE the bars (at bottom)
for i, (idx, row) in enumerate(leadtime_cancel.iterrows()):
    ax.text(i, 0.01, f"n={int(row['Total_Tickets']):,}", 
            ha='center', va='bottom', fontsize=10, weight='bold', 
            color='white', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, max(leadtime_cancel['Cancel_Rate']) * 1.2)

# Add more space at the bottom for labels
plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
plt.savefig(save_dir + 'leadtime_cancellation.png', dpi=300, bbox_inches='tight')
plt.show()