import json
import requests
import streamlit as st
import pandas as pd


# Read tracks dataset.
tracks_df = pd.read_csv('cleaned_tracks_dataset.csv')

st.title("Music Recommendations 🎵")
st.header("Tell us about your music preferences!")

with st.form("preferences_form"):
    track_ids = st.multiselect(
        "List some of your favorite bands or artists:",
        options=tracks_df["track_id"],
        help="Start typing to search for artists and select them from the list."
    )
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    st.toast("Fetching your music recommendations...")

    try:
        response = requests.post(
            "http://api:8000/recommend", 
            json={"track_ids": track_ids}
        )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")

        recommendations = response.json().get("recommendations", [])
        if recommendations:
            st.success("Here are some tracks we think you'll love!")
            for i, track in enumerate(recommendations, 1):
                st.write(f"{i}. {track['title']} by {track['artist']}")
        else:
            st.warning("We couldn't find any recommendations based on your preferences. Try adjusting them!")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
