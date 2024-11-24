from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.decomposition import NMF
import pandas as pd
import numpy as np

# Setup model.
nmf_model = NMF(n_components=10, init='nndsvd', random_state=42, max_iter=500)
# Read tracks dataset.
tracks_df = pd.read_csv('cleaned_tracks_dataset.csv')
# Compute embeddings.
track_factors = nmf_model.fit_transform(tracks_df)


class RecommendationRequest(BaseModel):
    track_ids: list[int]

class Recomendation(BaseModel):
    title: str
    artist: str

class RecommendationResponse(BaseModel):
    recommendations: List[Recomendation]

app = FastAPI(
    title="Music Recommendation API",
    version="0.1.0",
    description="An API to provide music recommendations based on user preferences."
)

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(data: RecommendationRequest) -> RecommendationResponse:
    scores = np.dot(track_factors[data.track_ids], track_factors.T).max(axis=0)
    recommendations = [(track, score) for track, score in zip(tracks_df.itertuples(), scores, strict=True)]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return RecommendationResponse(
        recommendations=[
            Recomendation(artist=str(track.artistname), title=str(track.trackname))
            for track, _ in recommendations[:3]
        ]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
