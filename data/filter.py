import pandas as pd

# Load the dataset
merged_data = pd.read_csv('merged_spotify_dataset.csv')

# Delete the 'user_id' column
if 'user_id' in merged_data.columns:
    merged_data = merged_data.drop(columns=['user_id'])

# Remove duplicate rows
merged_data = merged_data.drop_duplicates()

# Save the cleaned dataset to a new CSV file
merged_data.to_csv('cleaned_spotify_dataset.csv', index=False)

print("Successfully! The resulting dataset has been saved as 'cleaned_spotify_dataset.csv'.")
