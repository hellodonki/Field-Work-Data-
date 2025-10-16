import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

df = pd.read_csv("Ischnura_2000-2024.csv")
df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
df['Year'] = df['Datum'].dt.year
df['Parasite'] = pd.to_numeric(df['Parasite'], errors='coerce')
df = df[df['Parasite']> 0]
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')

df_hoje_a_6 = df[(df['Year']>=2000) & (df['Year']<=2024) & (df['Locale'] == 'Hoje_A_6')]
df_hoje_a_6 = df_hoje_a_6.dropna(subset=['Parasite', 'Length'])

# Group by year and calculate mean
mean_data = df_hoje_a_6.groupby('Year')[['Parasite', 'Length']].mean().reset_index()
print("Data length:", len(mean_data))
print(mean_data[['Parasite', 'Length']])

# Only run correlation if there are enough points
if len(mean_data) >= 2:
    r, p = pearsonr(mean_data['Parasite'], mean_data['Length'])
    print(f"Pearson r = {r:.3f}, p = {p:.3g}")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # First y-axis: Parasite
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Parasite (mean)', color=color)
    ax1.plot(mean_data['Year'], mean_data['Parasite'], marker='o', color=color, label='Parasite')
    ax1.tick_params(axis='y', labelcolor=color)

    # Second y-axis: Length
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Length (mean)', color=color)
    ax2.plot(mean_data['Year'], mean_data['Length'], marker='s', color=color, label='Length')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Median Parasite Count and Length in Hoje_A_6 (2000–2024)')
    fig.tight_layout()
    plt.grid(True)

    # Pearson correlation (based on yearly means)
    r, p = pearsonr(mean_data['Parasite'], mean_data['Length'])
    corr_text = f"Pearson r = {r:.3f}, p = {p:.4f}"

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Parasite axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Parasite (mean)', color=color1)
    ax1.plot(mean_data['Year'], mean_data['Parasite'], marker='o', color=color1, label='Parasite')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Length axis
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Length (mean)', color=color2)
    ax2.plot(mean_data['Year'], mean_data['Length'], marker='s', color=color2, label='Length')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add correlation text
    fig.text(0.05, 0.92, corr_text, fontsize=11, bbox=dict(facecolor='white', alpha=0.6))

    plt.title('Mean Parasite Count and Length in Hoje_A_6 (2000–2024)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data to calculate Pearson correlation.")


