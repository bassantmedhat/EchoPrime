import pandas as pd

# Load the second CSV file
csv2_path = "data/annotations-all.csv"
df2 = pd.read_csv(csv2_path)

# Rename Echo ID# to study_id for consistency
df2 = df2.rename(columns={"Echo ID#": "study_id"})

# Group data by study_id and aggregate the necessary columns
grouped = df2.groupby('study_id').agg({
    'path': lambda x: ', '.join(x),          # Concatenate file paths
    'view': lambda x: ', '.join(x),         # Concatenate views
    'as_label': lambda x: ', '.join(x),     # Concatenate AS labels
    'split': 'first'                        # Take the first split value (assumed consistent within a group)
}).reset_index()

# Rename columns to match the desired format
grouped.rename(columns={
    'study_id': 'study_id',
    'path': 'files',
    'view': 'views',
    'as_label': 'AS_label',
    'split': 'split'
}, inplace=True)

# Split into separate dataframes based on split
train_df = grouped[grouped['split'] == 'train']
test_df = grouped[grouped['split'] == 'test']
val_df = grouped[grouped['split'] == 'val']

# Save the separate CSV files
train_df.to_csv("data/AS_train.csv", index=False)
test_df.to_csv("data/AS_test.csv", index=False)
val_df.to_csv("data/AS_val.csv", index=False)

print("Train, test, and validation CSV files created successfully.")
