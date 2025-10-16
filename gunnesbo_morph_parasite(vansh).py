import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("gunnesbo_data.csv")

# Filter rows with non-null 'Morph' and 'Parasite' values
filtered_df = df[['Morph', 'Parasite']].dropna()

# Set the plot style
sns.set(style="whitegrid")

# Create the figure and plot
plt.figure(figsize=(10, 6))

# Box plot (summary stats only)
sns.boxplot(
    x="Morph", y="Parasite", data=filtered_df,
    width=0.3,
    showcaps=True,
    boxprops={'facecolor': 'None', 'edgecolor': 'black'},
    medianprops={'color': 'black'}
)

# Swarm plot (individual data points)
sns.swarmplot(
    x="Morph", y="Parasite", data=filtered_df,
    color=".25", size=3
)

# Customize labels and title
plt.title("Parasite Load by Morph Type (Box + Swarm Plot)")
plt.xlabel("Morph Type")
plt.ylabel("Parasite Count")
plt.xticks(rotation=15)
plt.tight_layout()

# Show the plot
plt.show()