import pandas as pd
import matplotlib.pyplot as plt

# Define thorax colors you're interested in (standardized to lowercase)
EXPECTED_THORAX_COLORS = sorted([
    'brown', 'blue', 'green', 'blue-green',
    'turquoise', 'violet-blue', 'violet-green', 'olive'
])

# Step 1: Load the data
file_path = "/Users/swastikmandal/Downloads/gunnesbo_data.csv"
try:
    df = pd.read_csv(file_path, sep=',')
    print(f"Successfully loaded data from: {file_path}")
    print(f"--- Raw DataFrame shape (rows, columns): {df.shape} ---")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Step 2: Clean column names
df.columns = df.columns.str.strip()

# Check required columns exist
if 'Locale' not in df.columns or 'Thor.col' not in df.columns:
    print("CRITICAL ERROR: Required columns 'Locale' or 'Thor.col' not found.")
    exit()

# Step 3: Prepare values
df['Locale_prepared'] = df['Locale'].astype(str).str.strip()
df['Thor.col_prepared'] = df['Thor.col'].astype(str).str.strip().str.lower()

print("\nUnique 'Locale_prepared' values:")
print(df['Locale_prepared'].value_counts(dropna=False))
print("\nUnique 'Thor.col_prepared' values:")
print(df['Thor.col_prepared'].value_counts(dropna=False))

# Step 4: Filter for 'Gunnesbo'
LOCALE_TO_FILTER = 'Gunnesbo'
df_target_locale_all_thor = df[df['Locale_prepared'] == LOCALE_TO_FILTER]

if df_target_locale_all_thor.empty:
    print(f"No data found for locale '{LOCALE_TO_FILTER}'")
    exit()

# Remove invalid/missing thorax colors
invalid_thor_strings = ['nan', '', 'na', 'n/a', 'none', 'unknown', 'missing']
df_target_locale_valid_thor = df_target_locale_all_thor[
    ~df_target_locale_all_thor['Thor.col_prepared'].isin(invalid_thor_strings)
]

if df_target_locale_valid_thor.empty:
    print(f"No valid thorax color data for locale '{LOCALE_TO_FILTER}'")
    exit()

# Step 5: Calculate probabilities
thorax_probs = df_target_locale_valid_thor['Thor.col_prepared'].value_counts(normalize=True).round(3)
all_plot_colors = sorted(set(EXPECTED_THORAX_COLORS + thorax_probs.index.tolist()))
thorax_probs_reindexed = thorax_probs.reindex(all_plot_colors, fill_value=0.0)

# Step 6: Print results
print(f"\n📍 Thorax Color Probabilities for Locale: {LOCALE_TO_FILTER}")
print(thorax_probs_reindexed[thorax_probs_reindexed > 0])

# Step 7: Plot bar chart
plot_data = thorax_probs_reindexed[thorax_probs_reindexed > 0]
if not plot_data.empty:
    plt.figure(figsize=(12, 8))
    bars = plt.bar(plot_data.index, plot_data.values, color='deepskyblue')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel("Thorax Color (lowercase standardized)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title(f"Thorax Color Probabilities in Locale: {LOCALE_TO_FILTER}", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, plot_data.max() * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Step 1: Load the data
    file_path = "/Users/swastikmandal/Downloads/gunnesbo_data.csv"
    df = pd.read_csv(file_path, sep=',')

    # Step 2: Clean column names
    df.columns = df.columns.str.strip()

    # Step 3: Check necessary columns
    required_cols = ['Locale', 'Morph', 'Parasite']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing one of the required columns: {required_cols}")
        exit()

    # Step 4: Prepare values
    df['Locale_clean'] = df['Locale'].astype(str).str.strip()
    df['Morph_clean'] = df['Morph'].astype(str).str.strip().str.lower()
    df['Parasite_clean'] = pd.to_numeric(df['Parasite'], errors='coerce')

    # Step 5: Normalize morph categories
    morph_map = {
        'violacea-androchrome': 'androchrome',
        'violacea infuscans': 'infuscans',
        'rufescens': 'obsoleta'
    }
    df['Morph_standard'] = df['Morph_clean'].replace(morph_map)

    # Step 6: Filter to Gunnesbo and drop rows with NaNs
    df_gunnesbo = df[df['Locale_clean'] == 'Gunnesbo']
    df_gunnesbo = df_gunnesbo.dropna(subset=['Morph_standard', 'Parasite_clean'])

    if df_gunnesbo.empty:
        print("No valid data for Gunnesbo with parasite and morph info.")
        exit()

    # Step 7: Group by morph and calculate mean parasite count
    morph_stats = df_gunnesbo.groupby('Morph_standard')['Parasite_clean'].agg(['count', 'mean']).sort_values('mean', ascending=False)

    print("\n📍 Mean Parasite Load per Morph in Gunnesbo:")
    print(morph_stats)

    # Step 8: Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=morph_stats.reset_index(),
        x='Morph_standard', y='mean',
        palette='viridis'
    )

    # Add value labels
    for i, val in enumerate(morph_stats['mean']):
        plt.text(i, val + 0.1, f"{val:.2f}", ha='center', fontsize=9)

    plt.title("Average Parasite Load per Morph in Gunnesbo", fontsize=14)
    plt.xlabel("Morph")
    plt.ylabel("Mean Parasite Count")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("No data to plot.")

