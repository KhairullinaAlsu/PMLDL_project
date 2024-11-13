import pandas as pd

# Load the datasets while skipping problematic rows
spotify_dataset = pd.read_csv('spotify_dataset.csv', on_bad_lines='skip')
spotify_songs = pd.read_csv('spotify_songs.csv')

spotify_dataset.rename(columns={' "trackname"': 'trackname', ' "artistname"': 'artistname'}, inplace=True)
spotify_songs.rename(columns={'track_name': 'trackname', 'track_artist': 'artistname'}, inplace=True)

# Merge datasets on 'trackname' and 'artistname' columns
merged_data = pd.merge(
    spotify_dataset[['user_id', 'trackname', 'artistname']],
    spotify_songs,
    on=['trackname', 'artistname'],
    how='inner'
)

# Drop duplicate tracks for the same user if needed
merged_data.drop_duplicates(subset=['user_id', 'trackname', 'artistname'], inplace=True)

# Save the resulting dataset to a new CSV file
merged_data.to_csv('merged_spotify_dataset.csv', index=False)

print("Dataset merged successfully! The resulting dataset has been saved as 'merged_spotify_dataset.csv'.")
