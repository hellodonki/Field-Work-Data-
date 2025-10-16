import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind # Import for t-test

# Configuration
LOCALE_TO_FILTER = 'Lomma'
GENDER_COLUMN_NAME = 'Sex'
# >>> GENDER MAPPING: 1 FOR MALES, 0 FOR FEMALES <<<
PARASITE_COLUMN_NAME = 'Parasite'
DATE_COLUMN_NAME = 'Datum'
LOCALE_COLUMN_NAME = 'Locale'
FILE_PATH = "Ischnura_2000-2024.csv"

def plot_parasite_by_gender_over_years_for_locale(
    df_input_locale: pd.DataFrame,
    gender_col: str,
    parasite_col: str,
    date_col: str,
    locale_name: str
):
    df = df_input_locale.copy()

    # --- 1. Data Validation and Further Cleaning ---
    if gender_col not in df.columns: # ... (error checks remain same)
        print(f"Error: Gender column '{gender_col}' not found for locale '{locale_name}'.")
        return
    if parasite_col not in df.columns:
        print(f"Error: Parasite column '{parasite_col}' not found for locale '{locale_name}'.")
        return
    if date_col not in df.columns:
        print(f"Error: Date column '{date_col}' not found for locale '{locale_name}'.")
        return

    df[parasite_col] = pd.to_numeric(df[parasite_col], errors='coerce')
    df[gender_col] = pd.to_numeric(df[gender_col], errors='coerce')
    df['Year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year

    df_cleaned = df.dropna(subset=[parasite_col, gender_col, 'Year'])
    df_cleaned = df_cleaned[df_cleaned[gender_col].isin([0, 1])]
    df_cleaned['Year'] = df_cleaned['Year'].astype(int)
    df_cleaned[gender_col] = df_cleaned[gender_col].astype(int)

    if df_cleaned.empty:
        print(f"No valid data (Parasite, Gender as 0/1, Year) for locale '{locale_name}' after cleaning.")
        return

    # --- Data for T-test: Use all valid individual observations ----
    male_parasites_all = df_cleaned[df_cleaned[gender_col] == 1][parasite_col].dropna()
    female_parasites_all = df_cleaned[df_cleaned[gender_col] == 0][parasite_col].dropna()
    
    p_value_text = "P-value: N/A (insufficient data for t-test)"
    ttest_result = None
    if len(male_parasites_all) >= 2 and len(female_parasites_all) >= 2: # Need at least 2 obs in each group for ttest_ind
        ttest_result = ttest_ind(male_parasites_all, female_parasites_all, equal_var=False, nan_policy='omit') # Welch's t-test
        p_value_text = f"Overall M vs F p-value: {ttest_result.pvalue:.3f}"
        print(f"T-test for {locale_name} ({parasite_col} between Males (1) and Females (0)):")
        print(f"  Statistic: {ttest_result.statistic:.2f}, P-value: {ttest_result.pvalue:.3f}")
        print(f"  Male N: {len(male_parasites_all)}, Mean: {male_parasites_all.mean():.2f}")
        print(f"  Female N: {len(female_parasites_all)}, Mean: {female_parasites_all.mean():.2f}")

    else:
        print(f"Insufficient data for t-test in locale '{locale_name}'. "
              f"Males: {len(male_parasites_all)}, Females: {len(female_parasites_all)} observations.")


    # --- 2. Calculate Mean Parasite Count per Year for Each Gender (for plotting lines) ---
    mean_parasite_per_year_gender = df_cleaned.groupby(['Year', gender_col])[parasite_col].mean().unstack(gender_col)
    
    column_rename_map = {1: 'Male_Avg_Parasite', 0: 'Female_Avg_Parasite'}
    mean_parasite_per_year_gender = mean_parasite_per_year_gender.rename(columns=column_rename_map)

    male_col_name = 'Male_Avg_Parasite'
    female_col_name = 'Female_Avg_Parasite'
    has_male_data = male_col_name in mean_parasite_per_year_gender.columns
    has_female_data = female_col_name in mean_parasite_per_year_gender.columns
    
    if not (has_male_data or has_female_data):
        print(f"No mean parasite data (for line plot) for locale '{locale_name}' after grouping.")
        return

    print(f"\n--- Mean Parasite Count by Gender Over Years for Locale: {locale_name} (for line plot) ---")
    print(mean_parasite_per_year_gender.head())

    # --- 3. Visualization ---
    plt.figure(figsize=(13, 7)) # Slightly wider for longer title

    if has_male_data:
        plt.plot(
            mean_parasite_per_year_gender.index,
            mean_parasite_per_year_gender[male_col_name],
            marker='o', linestyle='-', color='dodgerblue', label='Males (1)'
        )
    if has_female_data:
        plt.plot(
            mean_parasite_per_year_gender.index,
            mean_parasite_per_year_gender[female_col_name],
            marker='s', linestyle='--', color='orangered', label='Females (0)'
        )
        
    plt.xlabel("Year")
    plt.ylabel(f"Mean {PARASITE_COLUMN_NAME} Count")
    # Include p-value in the title
    plot_title = (f"Mean Parasite Count by Gender Over Years for Locale: {locale_name}\n"
                  f"{p_value_text}")
    plt.title(plot_title, fontsize=14)
    
    if not mean_parasite_per_year_gender.empty:
        all_years = sorted(mean_parasite_per_year_gender.index.unique().astype(int))
        if all_years:
            plt.xticks(ticks=all_years, rotation=45, ha="right")

    if has_male_data or has_female_data:
        plt.legend()
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to make room for the potentially longer title
    plt.show()

# --- Main script execution ---
try:
    df_main = pd.read_csv(FILE_PATH, sep=',')
    print(f"Successfully loaded data from: {FILE_PATH}")
except FileNotFoundError:
    print(f"CRITICAL ERROR: The file '{FILE_PATH}' was not found.")
    print("Using dummy data for demonstration purposes.")
    data_list = []
    locales = ['Lomma', 'Revinge', 'Lomma', 'Lomma'] # More Lomma data
    sexes = [1, 0] # Male=1, Female=0
    for year in range(2015, 2024):
        for _ in range(10): # More observations per year for better t-test in dummy data
            for loc in locales:
                for sex_val in sexes:
                    # Make a slight consistent difference for dummy t-test
                    base_parasite = 5 + (year - 2015) * 0.1
                    if sex_val == 1: # Male
                        parasite_count = base_parasite + np.random.randn() * 2 
                    else: # Female
                        parasite_count = base_parasite + 1.0 + np.random.randn() * 2 # Females slightly higher
                    
                    if loc == 'Lomma' and (pd.Series(range(10)).sample(1).iloc[0] > 0): # Ensure some valid data for Lomma
                         data_list.append({
                            'Datum': f'{year}-07-01', # Fixed month for simplicity in dummy
                            'Locale': loc,
                            'Sex': sex_val if pd.Series(range(10)).sample(1).iloc[0] > 0 else np.nan,
                            'Parasite': parasite_count if pd.Series(range(10)).sample(1).iloc[0] > 0 else np.nan,
                            'Length': 20 
                        })
    df_main = pd.DataFrame(data_list)
except Exception as e:
    print(f"CRITICAL ERROR: An error occurred while loading CSV: {e}")
    exit()

# (Rest of the main script execution remains the same)
df_main.columns = df_main.columns.str.strip()

if LOCALE_COLUMN_NAME not in df_main.columns:
    print(f"CRITICAL ERROR: Locale column '{LOCALE_COLUMN_NAME}' not found.")
    exit()

df_main[LOCALE_COLUMN_NAME] = df_main[LOCALE_COLUMN_NAME].astype(str).str.strip()
df_lomma = df_main[df_main[LOCALE_COLUMN_NAME] == LOCALE_TO_FILTER].copy()

if df_lomma.empty:
    print(f"\nNo data found for Locale: '{LOCALE_TO_FILTER}'. Cannot proceed.")
    print(f"Available unique values in '{LOCALE_COLUMN_NAME}' column: {df_main[LOCALE_COLUMN_NAME].unique()}")
else:
    print(f"\n--- Analyzing data specifically for Locale: {LOCALE_TO_FILTER} ---")
    print(f"Found {len(df_lomma)} records for '{LOCALE_TO_FILTER}'.")
    
    plot_parasite_by_gender_over_years_for_locale(
        df_lomma,
        GENDER_COLUMN_NAME,
        PARASITE_COLUMN_NAME,
        DATE_COLUMN_NAME,
        LOCALE_TO_FILTER
    )

print("\n--- Script Finished ---")