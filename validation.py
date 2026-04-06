import pandas as pd

csv1 = pd.read_csv("output.csv")
csv2 = pd.read_csv("output_parallel.csv")

sort_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveliness', 'valence']
sorted_csv1 = csv1.sort_values(by=sort_cols).reset_index(drop=True)
sorted_csv2 = csv2.sort_values(by=sort_cols).reset_index(drop=True)

# Vectorized comparison: creates a boolean mask (True/False) and sums the Trues
validation_amount = (sorted_csv1['c'] == sorted_csv2['c']).sum()
total_amount = len(sorted_csv1)

print(f"Validation amount: {validation_amount} out of {total_amount} for a total of {(validation_amount / total_amount) * 100:.2f}%")