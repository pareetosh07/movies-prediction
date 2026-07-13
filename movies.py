import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

# ---------------- CSS Styling ----------------
st.markdown(
    """
<style>
.stApp{ background-color:#FFF8F6; }
h1{ color:#E50914; text-align:center; font-weight:bold; }
.movie-card{ background:#FFFFFF; padding:18px; border-radius:12px; border:1px solid #F2DCDC; box-shadow:0px 2px 8px rgba(0,0,0,0.08); margin-top:15px; }
.stButton>button{ background:#E50914; color:white; border:none; border-radius:10px; height:45px; width:220px; font-size:18px; font-weight:bold; }
.stButton>button:hover{ background:#C40812; }
div[data-baseweb="select"]{ border-radius:10px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Banner ----------------
st.image(
    "https://m.media-amazon.com/images/I/616QXs8yg0L.png",
    use_container_width=True,
)
st.title("🎬 Movie Recommendation System")
st.write(
    "Select a movie you've watched and get **5 similar movie recommendations**."
)


# ---------------- Load Data & Calculate Similarity instantly ----------------
@st.cache_data
def load_data_and_similarity():
    # Your Direct Google Drive Download Link
    CSV_URL = "https://docs.google.com/uc?export=download&id=1KbLaNysPS6oiVo6z9iZiWnURBgNOUKo1"

    try:
        # 1. Download the dataframe directly from Google Drive
        df_loaded = pd.read_csv(CSV_URL)
        df_loaded.columns = df_loaded.columns.str.strip()
        df_loaded["title"] = df_loaded["title"].astype(str).str.strip()

        # 2. Find the correct text column automatically
        text_col = next(
            (
                col
                for col in ["tags", "Tags", "genres", "overview", "description"]
                if col in df_loaded.columns
            ),
            df_loaded.select_dtypes(include=["object"]).columns[-1],
        )

        # 3. Calculate similarity matrix in memory right now
        cv = CountVectorizer(max_features=10000, stop_words="english")
        dtm = cv.fit_transform(df_loaded[text_col].fillna(""))
        sim_matrix = cosine_similarity(dtm)

        return df_loaded, sim_matrix

    except Exception as e:
        st.error(
            f"❌ **Failed to load data or build engine.** Please verify your link permissions.\nError: {e}"
        )
        st.stop()


# Retrieve data and computed similarities instantly from cache
df, similarities = load_data_and_similarity()

# ---------------- Recommendation Setup ----------------
names = sorted(df["title"].unique())


def get_movie_index(name):
    try:
        return df[df["title"] == name].index[0]
    except IndexError:
        return -1


def get_movie_name(i):
    if 0 <= i < len(df):
        return df.loc[i, "title"]
    return ""


movie = st.selectbox("🍿 Select Movie", names)

if st.button("🎥 Recommend Movies"):
    index = get_movie_index(movie)
    if index == -1:
        st.error("Movie not found!")
    else:
        # Extract matches directly from memory
        similarity_index = sorted(
            list(enumerate(similarities[index])),
            key=lambda x: x[1],
            reverse=True,
        )

        st.subheader("✨ Recommended Movies")

        for i in range(1, 6):
            if i < len(similarity_index):
                recommended_name = get_movie_name(similarity_index[i][0])
                st.markdown(
                    f'<div class="movie-card"><h4>🎬 {i}. {recommended_name}</h4></div>',
                    unsafe_allow_html=True,
                )
