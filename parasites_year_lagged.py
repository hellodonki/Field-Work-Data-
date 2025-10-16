import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr # Import for correlation calculation

def plot_lagged_timeseries_single_plot_with_corr(
    df_input: pd.DataFrame,
    date_col: str,
    year_col_name: str = 'Year',
    locale_filter_col: str = 'Lomma',
    specific_locale: str = 'Lomma',
    var_x_col: str = 'Parasite',
    var_x_plus_1_col: str = 'Length',
    var_x_plot_label: str = 'Mean Value (Year X)',
    var_x_plus_1_plot_label: str = 'Mean Value (Year X+1)',
    figure_title_main: str = "Lagged Relationship Analysis"
):
    """
    Plots two time series on a single graph with dual y-axes, showing a lagged relationship,
    and calculates Pearson correlation between them.
    """
    # --- 1. Data Preparation ---
    df = df_input.copy()

    required_cols = [date_col, var_x_col, var_x_plus_1_col]
    if locale_filter_col:
        required_cols.append(locale_filter_col)
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in DataFrame.")
            return

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[year_col_name] = df[date_col].dt.year
    df[var_x_col] = pd.to_numeric(df[var_x_col], errors='coerce')
    df[var_x_plus_1_col] = pd.to_numeric(df[var_x_plus_1_col], errors='coerce')

    df_filtered = df.copy()
    filter_description = "All Data"
    if locale_filter_col and specific_locale:
        df_filtered[locale_filter_col] = df_filtered[locale_filter_col].astype(str).str.strip()
        df_filtered = df_filtered[df_filtered[locale_filter_col] == specific_locale]
        filter_description = f"Locale: {specific_locale}"
        if df_filtered.empty:
            print(f"No data found for locale '{specific_locale}' before processing.")
            # If you want to see what locales are available:
            # print(f"Available locales: {df_input[locale_filter_col].astype(str).str.strip().unique()}")
            return
    elif locale_filter_col and not specific_locale:
        print(f"Warning: 'locale_filter_col' provided without 'specific_locale'. No locale filtering.")

    df_filtered = df_filtered[
        df_filtered[year_col_name].notna() &
        df_filtered[var_x_col].notna() &
        df_filtered[var_x_plus_1_col].notna()
    ]

    if df_filtered.empty:
        print(f"No data for {var_x_col}/{var_x_plus_1_col} after filtering ({filter_description}) and NaN removal.")
        return

    # --- 2. Compute Means and Lag ---
    var_x_by_year = df_filtered.groupby(year_col_name)[var_x_col].mean().rename(f'Mean_{var_x_col}_X')
    var_x_plus_1_by_year = df_filtered.groupby(year_col_name)[var_x_plus_1_col].mean()
    var_x_plus_1_by_year_shifted = var_x_plus_1_by_year.shift(-1).rename(f'Mean_{var_x_plus_1_col}_X_plus_1')

    lagged_df = pd.concat([var_x_by_year, var_x_plus_1_by_year_shifted], axis=1).dropna()

    if lagged_df.empty:
        print(f"No data available in lagged_df after processing for {filter_description}.")
        print(f"  Mean {var_x_col} by Year (before lag):\n{var_x_by_year.head()}")
        print(f"  Mean {var_x_plus_1_col} by Year (before lag and shift):\n{var_x_plus_1_by_year.head()}")
        return

    # --- 3. Correlation Calculation ---
    correlation = None
    p_value = None
    correlation_text = "Not enough data points for correlation (need at least 2)."
    
    col_name_var_x = f'Mean_{var_x_col}_X'
    col_name_var_x_plus_1 = f'Mean_{var_x_plus_1_col}_X_plus_1'

    if len(lagged_df) >= 2:
        if col_name_var_x in lagged_df.columns and col_name_var_x_plus_1 in lagged_df.columns:
            correlation, p_value = pearsonr(lagged_df[col_name_var_x], lagged_df[col_name_var_x_plus_1])
            correlation_text = f"Pearson r: {correlation:.2f}, p-value: {p_value:.3f}"
            print(f"Correlation Analysis ({filter_description}): {correlation_text}")
        else:
            correlation_text = "Error: Columns for correlation not found in lagged_df."
            print(correlation_text)
    else:
        print(f"Correlation Analysis ({filter_description}): {correlation_text}")

    # --- 4. Plotting ---
    fig, ax1 = plt.subplots(figsize=(13, 7))

    color1 = 'mediumseagreen'
    ax1.set_xlabel("Year (X)")
    ax1.set_ylabel(f"{var_x_plot_label} ({var_x_col})", color=color1)
    line1 = ax1.plot(lagged_df.index, lagged_df[col_name_var_x], color=color1, marker='o', linestyle='-', label=f'Mean {var_x_col} (Year X)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2 = ax1.twinx()
    color2 = 'tomato'
    ax2.set_ylabel(f"{var_x_plus_1_plot_label} ({var_x_plus_1_col})", color=color2)
    line2 = ax2.plot(lagged_df.index, lagged_df[col_name_var_x_plus_1], color=color2, marker='s', linestyle='--', label=f'Mean {var_x_plus_1_col} (Year X+1)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    if not lagged_df.index.empty:
        tick_years = sorted(list(set(map(int, lagged_df.index.tolist()))))
        if tick_years:
            ax1.set_xticks(tick_years)
            ax1.set_xticklabels(tick_years, rotation=45, ha="right")

    title_string = f"{figure_title_main}: {var_x_col} (Year X) vs. {var_x_plus_1_col} (Year X+1)\n"
    title_string += f"{filter_description} | {correlation_text}"
    plt.title(title_string, fontsize=14)

    fig.tight_layout()
    plt.show()

    print("\nLagged DataFrame used for plotting and correlation:")
    print(lagged_df)

    # --- Load your data (or use dummy data) ---
try:
    df_main = pd.read_csv("Ischnura_2000-2024.csv", low_memory=False)
    print("Successfully loaded 'Ischnura_2000-2024.csv'")
except FileNotFoundError:
    print("Warning: 'Ischnura_2000-2024.csv' not found. Using dummy data for demonstration.")
    data_list = []
    locales = ['Lomma'] # Added Lomma
    for year_val in range(2000, 2025):
        for month_val in [5, 6, 7, 8]:
            for loc_idx, current_locale in enumerate(locales):
                data_list.append({
                    'Datum': f'{year_val}-{month_val:02d}-15',
                    'Locale': current_locale,
                    'Parasite': (5 + loc_idx*2) + (year_val % (7-loc_idx)) * 0.8 + month_val * 0.5 + ((year_val + month_val) % 3 - 1) * 1.5 + (year_val-2000)* (0.1 - loc_idx*0.03),
                    'Length': (25 - loc_idx*1.5) - (year_val % (5+loc_idx)) * 0.7 + month_val * 0.3 + ((year_val + month_val) % 4 - 1.5) * 1.2 - (year_val-2000)* (0.05 - loc_idx*0.01),
                })
    df_main = pd.DataFrame(data_list)
    # Introduce some NaNs
    df_main.loc[df_main.sample(frac=0.03).index, 'Parasite'] = pd.NA
    df_main.loc[df_main.sample(frac=0.03).index, 'Length'] = pd.NA


# --- Call the function for 'Lomma' ---
if 'Locale' in df_main.columns and 'Lomma' in df_main['Locale'].astype(str).str.strip().unique():
    print("\n--- Plotting for Locale: Lomma ---")
    plot_lagged_timeseries_single_plot_with_corr(
        df_input=df_main,
        date_col='Datum',
        locale_filter_col='Locale',
        specific_locale='Lomma',      # <<<< Key change here
        var_x_col='Parasite',
        var_x_plus_1_col='Length',
        var_x_plot_label='Mean Parasite Count',
        var_x_plus_1_plot_label='Mean Body Length (mm)',
        figure_title_main="Lagged Analysis: Parasites & Body Length"
    )
elif 'Locale' not in df_main.columns:
    print("Error: 'Locale' column not found in the DataFrame. Cannot filter for 'Lomma'.")
else:
    print(f"Warning: Locale 'Lomma' not found in the 'Locale' column. Cannot generate the specific plot.")
    print(f"Available locales: {df_main['Locale'].astype(str).str.strip().unique()}")