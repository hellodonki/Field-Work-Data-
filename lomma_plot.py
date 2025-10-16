import pandas as pd

#Load the full dataset
df = pd.read_csv("lomma_data.csv")

df = df.dropna(subset=['Parasite', 'Length'])
df = df[df['Length']>10]
df = df[df['Parasite']>1]

import matplotlib.pyplot as plt
plt.figure(figsize = (8,6))
plt.scatter(df['Parasite'], df['Length'], alpha = 0.4, color='red')
plt.xlabel('Parasite Count')
plt.ylabel('Body length')
plt.grid(True, linestyle = '--', color = 'gray')
plt.show()