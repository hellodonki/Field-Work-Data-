import pandas as pd

#Load the full dataset
df = pd.read_csv("Ischnura_2000-2024.csv")

# Define 'gunnesbo_data' by filtering the rows where Locale is 'gunnesbo'
lomma_data = df[df['Locale'] == 'Lomma']


# Check again what values exist
print(df['Locale'].unique())

# Now filter for 'gunnesbo'
lomma_data = df[df['Locale'] == 'Lomma']

# Check how many rows you got
print(f"Rows found: {lomma_data.shape[0]}")

# Optional: Save it to a CSV file
lomma_data.to_csv("lomma_data.csv", index=False)
lomma_data.to_csv("lomma_data.csv", index=False, sep=',')