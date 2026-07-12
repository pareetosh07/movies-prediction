import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

# ---------------- CSS ----------------
st.markdown(
    """
<style>
.stApp{
    background-color:#FFF8F6;
}
h1{
    color:#E50914;
    text-align:center;
    font-weight:bold;
}
.movie-card{
    background:#FFFFFF;
    padding:18px;
    border-radius:12px;
    border:1px solid #F2DCDC;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    margin-top:15px;
}
.stButton>button{
    background:#E50914;
    color:white;
    border:none;
    border-radius:10px;
    height:45px;
    width:220px;
    font-size:18px;
    font-weight:bold;
}
.stButton>button:hover{
    background:#C40812;
}
div[data-baseweb="select"]{
    border-radius:10px;
}
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

# ---------------- Load Data Safely ----------------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "data.csv")  # Target correct filename

    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    fallback_path = r"C:\movies predict\final_data.csv"
    if os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)

    st.error(
        f"❌ **Data file not found!** Please make sure **`data.csv`** is saved correctly inside: `{script_dir}`"
    )
    st.stop()


# Load the data frame
df = load_data()

# Clean column names (removes leading/trailing spaces)
df.columns = df.columns.str.strip()

# Validate the title column
if "title" in df.columns:
    df["title"] = df["title"].astype(str).str.strip()
else:
    st.error(
        f"❌ Your CSV file is missing a **'title'** column. Available columns: {list(df.columns)}"
    )
    st.stop()

# Auto-detect a text column to use for recommendations (tags, genres, overview, etc.)
text_column = None
for col in ["tags", "Tags", "genres", "overview", "description"]:
    if col in df.columns:
        text_column = col
        break

if not text_column:
    text_column = df.select_dtypes(include=["object"]).columns[-1]

# ---------------- Similarity Matrix Path Setup ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(script_dir, "similarities.pkl")

# ---------------- Similarity Matrix Generation Guard ----------------
if not os.path.exists(pkl_path):
    st.warning("⚠️ Similarity matrix not found. Please generate it first.")
    st.info(f"💡 Using column **'{text_column}'** to analyze similarity.")
    if st.button("🔄 Generate Similarity Matrix"):
        with st.spinner("Generating matrix... Please wait (this takes a moment)"):
            cv = CountVectorizer(max_features=10000, stop_words="english")
            dtm = cv.fit_transform(df[text_column].fillna(""))
            similarities = cosine_similarity(dtm)
            with open(pkl_path, "wb") as f:
                pickle.dump(similarities, f)
        st.success("Similarity Matrix Generated Successfully!")
        st.rerun()
    st.stop()

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

# ---------------- Recommendation Action ----------------
if st.button("🎥 Recommend Movies"):
    index = get_movie_index(movie)

    if index == -1:
        st.error("Movie not found!")
    else:
        with open(pkl_path, "rb") as f:
            similarities = pickle.load(f)

        similarity_index = list(enumerate(similarities[index]))
        similarity_index = sorted(
            similarity_index, key=lambda x: x[1], reverse=True
        )

        st.subheader("✨ Recommended Movies")

        for i in range(1, 6):
            if i < len(similarity_index):
                recommended_name = get_movie_name(similarity_index[i][0])
                st.markdown(
                    f"""
                <div class="movie-card">
                <h4>🎬 {i}. {recommended_name}</h4>
                </div>
                """,
                    unsafe_allow_html=True,
                )