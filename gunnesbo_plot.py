import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('gunnesbo_data.csv')

# Clean column names
#df.columns = df.columns.str.strip().str.lower()

# Confirm columns
print(df.columns.tolist())


# Drop any remaining invalid entries
df = df.dropna(subset=['Parasite', 'Length'])
df = df[df['Length']>10]
df = df[df['Parasite']>0]

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df["Length"], df["Parasite"],
            alpha=0.7, color='mediumseagreen', edgecolors='black')
plt.xlabel('Body Length (mm)')
plt.ylabel('Parasite Count')
plt.title('Parasite Load vs Body Length (Gunnesbo)')
plt.grid(True)
plt.tight_layout()
plt.show()