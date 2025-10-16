import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('lomma_data.csv')
df['Datum'] = pd.to_datetime(df['Datum'], errors = 'coerce')
df['Year'] = df['Datum'].dt.year 


#to filter to find years between 2016 & 2024
df_lomma = df[(df['Locale'] == 'Lomma') & (df['Year'] >= 2016) & (df['Year'] <= 2024)]
df_lomma['Parasite'] = pd.to_numeric(df_lomma['Parasite'], errors = 'coerce')
df_lomma = df_lomma[df_lomma['Parasite']>0]
df_lomma = df_lomma.dropna(subset = ['Parasite'])

plt.figure(figsize=(10,6))
sns.boxplot(df_lomma, x = 'Year', y = 'Parasite')
plt.title('Parasite distribution in Lomma between 2016 & 2024')
plt.xlabel('Year')
plt.ylabel('Number of Parasites')
plt.grid(True)
plt.tight_layout()
plt.show()
