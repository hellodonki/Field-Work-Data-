import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import kruskal
import itertools

# Step 1: Load data (semicolon-separated)
df = pd.read_csv("Ischnura_2000-2024.csv", sep=',')

# Step 2: Clean column names
df.columns = df.columns.str.strip()

# Step 3: Standardize Morph names
df['Morph'] = df['Morph'].str.strip()

# Replace complex Morph names
df['Morph'] = df['Morph'].replace({
    'violacea-androchrome': 'androchrome',
    'violacea-infuscans': 'infuscans',
    'rufescens': 'obsoleta'
})

# Step 4: Convert relevant columns to numeric
df['Parasite'] = pd.to_numeric(df['Parasite'], errors='coerce')
df = df[df['Parasite']>0]
order = df.groupby('Morph')['Parasite'].median().sort_values(ascending=False).index

# Step 5: Drop NA values
df_clean = df.dropna(subset=['Morph', 'Parasite'])

mean_values = df_clean.groupby('Morph')['Parasite'].mean().round(4)

groups = [group['Parasite'] for name, group in df_clean.groupby('Morph')]
stat, p_kruskal = kruskal(*groups)
print(f"Kruskal-Wallis p = {p_kruskal:.4e}")

# Step 2: Dunn’s post hoc test
posthoc = sp.posthoc_dunn(df_clean, val_col='Parasite', group_col='Morph', p_adjust='bonferroni')

# Step 3: Plot boxplot
order = df_clean.groupby('Morph')['Parasite'].median().sort_values(ascending=False).index.tolist()

plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_clean, x='Morph', y='Parasite', order=order, palette='Set3')
plt.title("Parasite Load by Morph Type with Significance Annotations")
plt.xlabel("Morph")
plt.ylabel("Parasite Count")

# Step 4: Add significance stars
def get_significance_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''

# Position settings
y_max = df_clean['Parasite'].max()
height = y_max * 0.05  # spacing between significance lines
num_lines = 0

for (i, morph1), (j, morph2) in itertools.combinations(enumerate(order), 2):
    p_val = posthoc.loc[morph1, morph2]
    star = get_significance_stars(p_val)
    if star:
        x1, x2 = i, j
        y = y_max + height * num_lines
        h = height / 2
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.3, color='black')
        ax.text((x1 + x2) / 2, y + h, star, ha='center', va='bottom', color='black', fontsize=12)
        num_lines += 1

plt.tight_layout()
plt.show()