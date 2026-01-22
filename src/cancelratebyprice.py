import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Carica il dataset
df = pd.read_csv('./data/updated_data.csv')

# Definisci i bin per il prezzo
bins = [0, 11, 12, 13, 14, 15, 16, 17, float('inf')]
labels = ['0-11', '11-12', '12-13', '13-14','14-15', '15-16', '16-17', '17+']

# Crea una nuova colonna con i range di prezzo
df['Price_Range'] = pd.cut(df['LogPrice'], bins=bins, labels=labels, right=False)

# ... (loading and binning code remains the same) ...

# 1. Group and calculate
cancel_stats = df.groupby('Price_Range', observed=False)['Cancel'].agg(
    cancel_rate='mean',
    total_cancellations='sum',
    total_tickets='count'
).reset_index()

# Note: We don't sort by cancel_rate here so we can see the PRICE TREND
plt.figure(figsize=(12, 6))

# 2. Create the barplot
# Assigning x to hue avoids the palette warning in newer Seaborn versions
ax = sns.barplot(
    x='Price_Range', 
    y='cancel_rate', 
    data=cancel_stats, 
    palette='viridis',
    hue='Price_Range',
    legend=False
)

# 3. Improved Annotation
# Using the bar's height directly ensures the label matches the bar
for p in ax.patches:
    height = p.get_height()
    # Find the corresponding total_cancellations from the dataframe
    # This is safer than zipping if the plot order ever changes
    ax.annotate(
        f'{height:.2%}', # Shows the percentage rate
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom', xytext=(0, 5),
        textcoords='offset points'
    )
for i, (idx, row) in enumerate(cancel_stats.iterrows()):
    ax.text(i, 0.01, f"n={int(row['total_tickets']):,} \n c={int(row['total_cancellations']):,}", 
            ha='center', va='bottom', fontsize=10, weight='bold', 
            color='white', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

legend_labels = [
    mpatches.Patch(color='none', label='c = number of cancelled tickets'),
    mpatches.Patch(color='none', label='n = total tickets sold')
]
plt.legend(handles=legend_labels, loc='upper right', frameon=True, fontsize=8)

plt.title('Cancellation Rate by Price Range')
plt.ylabel('Cancellation Rate (Mean)')
plt.show()
