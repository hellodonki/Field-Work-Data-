import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp

# Step 1: Load and clean data
df = pd.read_csv("Ischnura_2000-2024.csv", low_memory=False)
df['Parasite'] = pd.to_numeric(df['Parasite'], errors='coerce')
df = df[df['Parasite'] > 0]
df['Age'] = df['Age'].astype(str).str.strip()
df_clean = df.dropna(subset=['Age', 'Parasite'])

# Step 2: Kruskal–Wallis test (overall difference)
groups = [group['Parasite'] for _, group in df_clean.groupby('Age')]
stat, p_kw = kruskal(*groups)

print(f"Kruskal–Wallis H = {stat:.3f}")
print(f"Kruskal–Wallis p = {p_kw:.4e}")
if p_kw < 0.05:
    print("→ Significant difference in parasite load between age groups.")
else:
    print("→ No significant difference found.")

# Step 3: Dunn’s test for pairwise comparisons (Bonferroni corrected)
posthoc = sp.posthoc_dunn(df_clean, val_col='Parasite', group_col='Age', p_adjust='bonferroni')

# Step 4: Heatmap of Dunn’s test p-values
plt.figure(figsize=(10, 8))
sns.heatmap(posthoc, annot=True, fmt=".3f", cmap='coolwarm', cbar_kws={'label': 'p-value'})
plt.title("Dunn's Test: Pairwise Comparison of Parasite Load Across Age Groups")
plt.tight_layout()
plt.show()