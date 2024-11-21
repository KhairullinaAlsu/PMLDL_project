import pandas as pd
import numpy as np

# Data loading: Load the favorite tracks for each user and the model recommendations from CSV files
favorite_tracks_df = pd.read_csv('user_favorite_tracks.csv') 

model_predictions = {
    'GNN': pd.read_csv('user_recommendations_GNN.csv'),
    'KNN': pd.read_csv('user_recommendations_KNN.csv'),
    'NMF': pd.read_csv('user_recommendations_NMF.csv'),
    'LightFM': pd.read_csv('user_recommendations_LightFM.csv')
}

# Function to calculate Mean Reciprocal Rank (MRR) for a single model
def calculate_mrr(model_df, favorite_tracks_df):
    mrr = 0  # Initialize MRR sum
    user_count = 0  # Counter for the number of users with recommendations

    # Iterate through each user in the favorite tracks dataset
    for user_id in favorite_tracks_df['user']:
        # Skip if user_id is not found in the model's recommendations
        if user_id not in model_df['user'].values:
            continue  

        # Get the list of recommended tracks for the user
        recommendations = model_df[model_df['user'] == user_id].iloc[0, 1:].values  
        # Get the user's list of favorite tracks (last 10 favorites)
        favorite_tracks = favorite_tracks_df[favorite_tracks_df['user'] == user_id].iloc[0, -10:].values

        # Initialize reciprocal rank
        reciprocal_rank = 0
        # Check each recommended track's rank
        for rank, rec_track in enumerate(recommendations, start=1):
            # If the recommended track is in the user's favorite list, set reciprocal rank and break
            if rec_track in favorite_tracks:
                reciprocal_rank = 1 / rank
                break

        # Add the user's reciprocal rank to MRR sum and increase user count
        mrr += reciprocal_rank
        user_count += 1

    # Calculate and return the average MRR for all users
    return mrr / user_count if user_count > 0 else 0

# Calculate and print MRR for each recommendation model
for model_name, predictions_df in model_predictions.items():
    mrr = calculate_mrr(predictions_df, favorite_tracks_df)
    print(f"MRR for model {model_name}: {mrr:.4f}")
