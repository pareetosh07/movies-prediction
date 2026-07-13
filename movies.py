import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

# ---------------- CSS Styling ----------------
st.markdown(
    """
<style>
.stApp { background-color: #FFF8F6; }
h1 { color: #E50914; text-align: center; font-weight: bold; }
.movie-card { background: #FFFFFF; padding: 18px; border-radius: 12px; border: 1px solid #F2DCDC; box-shadow: 0px 2px 8px rgba(0,0,0,0.08); margin-top: 15px; }
.stButton>button { background: #E50914; color: white; border: none; border-radius: 10px; height: 45px; width: 220px; font-size: 18px; font-weight: bold; }
.stButton>button:hover { background: #C40812; }
div[data-baseweb="select"] { border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

st.image(
    "https://m.media-amazon.com/images/I/616QXs8yg0L.png",
    use_container_width=True,
)
st.title("🎬 Movie Recommendation System")

# ---------------- Automated Download & Auto-Pickling ----------------

PKL_FILE = "similarity_matrix.pkl"
DATA_URL = "https://raw.githubusercontent.com/rehab-f/Movie-Recommendation-System/main/tmdb_5000_movies.csv"


@st.cache_resource
def auto_train_and_pickle():
    # If the .pkl file already exists on the server, just load it instantly
    if os.path.exists(PKL_FILE):
        with open(PKL_FILE, "rb") as f:
            df_loaded, sim_matrix = pickle.load(f)
        return df_loaded, sim_matrix

    # Otherwise, automatically fetch the data, calculate, and create the .pkl file
    with st.spinner("⏳ First time initialization: Training model and creating .pkl file..."):
        try:
            df_loaded = pd.read_csv(DATA_URL)
            df_loaded.columns = df_loaded.columns.str.strip()

            # Ensure 'title' column exists
            if (
                "title" not in df_loaded.columns
                and "original_title" in df_loaded.columns
            ):
                df_loaded.rename(
                    columns={"original_title": "title"}, inplace=True
                )

            df_loaded["title"] = df_loaded["title"].astype(str).str.strip()

            # Combine structural text tags for processing
            df_loaded["overview"] = df_loaded["overview"].fillna("")
            df_loaded["genres"] = df_loaded["genres"].fillna("")
            df_loaded["tags"] = (
                df_loaded["overview"] + " " + df_loaded["genres"]
            )

            # Train the vectorizer and compute similarity matrix
            cv = CountVectorizer(max_features=5000, stop_words="english")
            dtm = cv.fit_transform(df_loaded["tags"])
            sim_matrix = cosine_similarity(dtm)

            # AUTOMATICALLY DUMP TO .PKL FILE ON STREAMLIT SERVER
            with open(PKL_FILE, "wb") as f:
                pickle.dump((df_loaded, sim_matrix), f)

            return df_loaded, sim_matrix

        except Exception as e:
            st.error(f"❌ Failed to auto-generate engine: {e}")
            st.stop()


# Run the automated routine
df, similarities = auto_train_and_pickle()

# ---------------- Recommendation UI ----------------
names = sorted(df["title"].unique())
movie = st.selectbox("🍿 Select Movie", names)

if st.button("🎥 Recommend Movies"):
    try:
        index = df[df["title"] == movie].index[0]
        similarity_index = sorted(
            list(enumerate(similarities[index])),
            key=lambda x: x[1],
            reverse=True,
        )

        st.subheader("✨ Recommended Movies")
        count = 0
        for i in range(1, len(similarity_index)):
            if count >= 5:
                break
            rec_name = df.iloc[similarity_index[i][0]]["title"]
            if rec_name != movie:
                st.markdown(
                    f'<div class="movie-card"><h4>🎬 {count+1}. {rec_name}</h4></div>',
                    unsafe_allow_html=True,
                )
                count += 1
    except Exception:
        st.error("Could not fetch recommendations for this title.")
