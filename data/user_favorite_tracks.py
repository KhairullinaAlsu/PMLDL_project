import pandas as pd

# Load the merged dataset
merged_data = pd.read_csv('merged_spotify_dataset.csv')

# Create a dictionary where each user has exactly 20 favorite tracks
favorite_tracks_dict = {
    user_id: tracks[:20]  # Limit to 10 tracks if more are present
    for user_id, tracks in merged_data.groupby('user_id')['trackname'].apply(list).items()
    if len(tracks) >= 20  # Only include users with 10 or more tracks
}

# Convert the dictionary into a DataFrame
favorite_tracks_df = pd.DataFrame.from_dict(favorite_tracks_dict, orient='index', columns=[f'favorite_song_{i+1}' for i in range(20)])
favorite_tracks_df.reset_index(inplace=True)
favorite_tracks_df.rename(columns={'index': 'user'}, inplace=True)

# Save the DataFrame to a CSV file
favorite_tracks_df.to_csv('user_favorite_tracks.csv', index=False)

print("CSV file 'user_favorite_tracks.csv' created successfully with columns user, favorite_song_1, ..., favorite_song_20.")
