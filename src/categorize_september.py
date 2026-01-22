import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# load dataset
df = pd.read_csv('./data/updated_data.csv')

# filters data to get only september (column 'MonthDeparture')
df_settembre = df[df['MonthDeparture'] == 9]

# cathegorize lead time days for september in order to save data
print(df_settembre)
def categorizza_settimane(days):
    if pd.isna(days):
        return 'N/A'
    settimane = math.ceil(days / 7)
    return f'{settimane} settimane'

df_settembre['LeadTime_Settimane'] = df_settembre['LeadTime_Days'].apply(categorizza_settimane)

# calculates cancellation rates for every category of LeadTime_Settimane
tasso_cancellazione = df_settembre.groupby('LeadTime_Settimane')['Cancel'].agg(
    tasso_cancellazione=('mean')
).reset_index()

# order by decreasing cancellation rate
tasso_cancellazione = tasso_cancellazione.sort_values(by='tasso_cancellazione', ascending=False)


print(tasso_cancellazione)


plt.figure(figsize=(10, 6))
sns.barplot(
    x='LeadTime_Settimane',
    y='tasso_cancellazione',
    data=tasso_cancellazione,
    palette='viridis'
)
plt.title('Tasso di cancellazione per Lead Time (Settimane) - Settembre')
plt.xlabel('Lead Time (Settimane)')
plt.ylabel('Tasso di Cancellazione')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
