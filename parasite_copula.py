import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu # For statistical tests

# --- Configuration ---
LOCALE_TO_FILTER = 'Lomma'      # Set to None if you want to analyze the entire dataset, or a specific locale name
COPULA_COLUMN_NAME = 'Copula'   # <--- !!! UPDATE THIS if your copula column name is different !!!
PARASITE_COLUMN_NAME = 'Parasite'# <--- !!! UPDATE THIS if your parasite count column name is different !!!
DATE_COLUMN_NAME = 'Datum'      # <--- !!! UPDATE THIS if your date column for extracting year is different !!!
LOCALE_COLUMN_NAME = 'Locale'   # <--- !!! UPDATE THIS if your locale column has a different name !!!
FILE_PATH = "Ischnura_2000-2024.csv" # <--- !!! UPDATE THIS to your actual file path !!!
SIGNIFICANCE_LEVEL = 0.05       # Alpha for hypothesis testing

def plot_parasite_by_copula_over_years(
    df_input: pd.DataFrame,
    copula_col: str,
    parasite_col: str,
    date_col: str,
    plot_title_suffix: str
):
    df = df_input.copy()

    # --- 1. Data Validation and Further Cleaning ---
    if copula_col not in df.columns:
        print(f"Error: Copula column '{copula_col}' not found.")
        return
    if parasite_col not in df.columns:
        print(f"Error: Parasite column '{parasite_col}' not found.")
        return
    if date_col not in df.columns:
        print(f"Error: Date column '{date_col}' not found.")
        return

    df[parasite_col] = pd.to_numeric(df[parasite_col], errors='coerce')
    df[copula_col] = pd.to_numeric(df[copula_col], errors='coerce')
    df['Year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year

    df_cleaned = df.dropna(subset=[parasite_col, copula_col, 'Year'])
    df_cleaned = df_cleaned[df_cleaned[copula_col].isin([0, 1])]
    df_cleaned['Year'] = df_cleaned['Year'].astype(int)
    df_cleaned[copula_col] = df_cleaned[copula_col].astype(int)

    if df_cleaned.empty:
        print(f"No valid data (Parasite, Copula as 0/1, Year) {plot_title_suffix} after cleaning.")
        return

    # --- 2. Calculate Mean Parasite Count per Year for Each Copula Status ---
    mean_parasite_per_year_copula = df_cleaned.groupby(['Year', copula_col])[parasite_col].mean().unstack(copula_col)
    
    column_rename_map = {0: 'Copula_0_Avg_Parasite', 1: 'Copula_1_Avg_Parasite'}
    mean_parasite_per_year_copula = mean_parasite_per_year_copula.rename(columns=column_rename_map)

    copula0_col_name = 'Copula_0_Avg_Parasite'
    copula1_col_name = 'Copula_1_Avg_Parasite'

    has_copula0_data_ts = copula0_col_name in mean_parasite_per_year_copula.columns
    has_copula1_data_ts = copula1_col_name in mean_parasite_per_year_copula.columns

    if not (has_copula0_data_ts or has_copula1_data_ts):
        print(f"No mean parasite data to plot {plot_title_suffix} after grouping by year and copula status.")
        return

    print(f"\n--- Mean Parasite Count by Copula Status Over Years {plot_title_suffix} ---")
    print(mean_parasite_per_year_copula.head())

    # --- 3. Overall Statistical Test ---
    parasites_copula0 = df_cleaned[df_cleaned[copula_col] == 0][parasite_col].dropna()
    parasites_copula1 = df_cleaned[df_cleaned[copula_col] == 1][parasite_col].dropna()
    
    p_value_overall_str = "N/A (insufficient data for test)"
    test_used_overall = ""

    if len(parasites_copula0) >= 5 and len(parasites_copula1) >= 5: # Min sample size for a basic test
        try:
            stat, p_val = mannwhitneyu(parasites_copula0, parasites_copula1, alternative='two-sided', nan_policy='omit')
            test_used_overall = "Mann-Whitney U"
            p_value_overall_str = f"Overall p-value ({test_used_overall}): {p_val:.3f}"
            if p_val < SIGNIFICANCE_LEVEL:
                p_value_overall_str += f" (Significant at alpha={SIGNIFICANCE_LEVEL})"
            else:
                p_value_overall_str += f" (Not significant at alpha={SIGNIFICANCE_LEVEL})"
        except ValueError as e: 
             p_value_overall_str = f"Statistical test error: {e}"
    else:
        print(f"Not enough data in one or both copula groups for an overall statistical test {plot_title_suffix}.")

    print(f"Overall comparison of parasite loads {plot_title_suffix}: {p_value_overall_str}")

    # --- 4. Visualization ---
    plt.figure(figsize=(13, 7)) 

    if has_copula0_data_ts:
        plt.plot(
            mean_parasite_per_year_copula.index,
            mean_parasite_per_year_copula[copula0_col_name],
            marker='o', linestyle='-', color='forestgreen', label='Copula = 0 (Avg Parasite)'
        )
    if has_copula1_data_ts:
        plt.plot(
            mean_parasite_per_year_copula.index,
            mean_parasite_per_year_copula[copula1_col_name],
            marker='s', linestyle='--', color='purple', label='Copula = 1 (Avg Parasite)'
        )
        
    plt.xlabel("Year")
    plt.ylabel(f"Mean {PARASITE_COLUMN_NAME} Count")
    title = f"Mean Parasite Count by Copula Status Over Years {plot_title_suffix}\n{p_value_overall_str}"
    plt.title(title, fontsize=12)
    
    if not mean_parasite_per_year_copula.empty:
        all_years = sorted(mean_parasite_per_year_copula.index.unique().astype(int))
        if all_years:
            plt.xticks(ticks=all_years, rotation=45, ha="right")

    if has_copula0_data_ts or has_copula1_data_ts:
        plt.legend()
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Main script execution ---
try:
    df_main = pd.read_csv(FILE_PATH, sep=',') # Use low_memory=False if you have mixed types and large file
    print(f"Successfully loaded data from: {FILE_PATH}")
    print(f"Initial DataFrame shape: {df_main.shape}")
except FileNotFoundError:
    print(f"CRITICAL ERROR: The file '{FILE_PATH}' was not found.")
    print("Ensure the file path is correct and the file is in the specified location.")


except Exception as e:
    print(f"CRITICAL ERROR: An error occurred while loading CSV: {e}")
    exit()

# Clean column names from whitespace
df_main.columns = df_main.columns.str.strip()

# --- Filtering Logic: This part prepares `data_to_plot` and `title_suffix` ---
data_to_plot = df_main.copy()  # Start with the full dataframe
title_suffix = "(Overall Data)" # Default title suffix


if data_to_plot.empty:
    print(f"Data to plot is empty for {title_suffix}. Halting.")
else:
    print(f"\n--- Analyzing data {title_suffix} ---")
    print(f"Found {len(data_to_plot)} records for analysis.")
    
    # Corrected function call:
    plot_parasite_by_copula_over_years(
        data_to_plot,         # 1st arg: the DataFrame
        COPULA_COLUMN_NAME,   # 2nd arg: copula_col string
        PARASITE_COLUMN_NAME, # 3rd arg: parasite_col string
        DATE_COLUMN_NAME,     # 4th arg: date_col string
        title_suffix          # 5th arg: plot_title_suffix string
    )

print("\n--- Script Finished ---")