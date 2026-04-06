import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load and prepare data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df = df[['title', 'type', 'listed_in', 'description', 'director', 'cast', 'country']].copy()

    for col in ['listed_in', 'description', 'director', 'cast', 'country']:
        df[col] = df[col].fillna('')

    df['combined_features'] = (
        df['listed_in'] + ' ' +
        df['description'] + ' ' +
        df['director'] + ' ' +
        df['cast'] + ' ' +
        df['country']
    )

    df = df.drop_duplicates(subset='title').reset_index(drop=True)
    return df

@st.cache_resource
def build_model(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(dataframe.index, index=dataframe['title'].str.lower()).drop_duplicates()
    return cosine_sim, indices

df = load_data()
cosine_sim, indices = build_model(df)

# -----------------------------
# Recommendation function
# -----------------------------
def recommend(title, num_recommendations=5, content_type="All"):
    title_lower = title.lower()

    if title_lower not in indices:
        return pd.DataFrame()

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    similar_indices = [i[0] for i in sim_scores[1:30]]
    recs = df.iloc[similar_indices][['title', 'type', 'listed_in', 'description']].copy()

    if content_type != "All":
        recs = recs[recs['type'] == content_type]

    return recs.head(num_recommendations).reset_index(drop=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎬 About")
st.sidebar.markdown("""
This project is a **Content-Based Recommendation System**.

### 🔍 How it works:
- Uses **TF-IDF Vectorization**
- Computes **Cosine Similarity**
- Recommends similar titles based on:
  - Genre
  - Description
  - Cast
  - Director
  - Country

### ⚙️ Tech Stack:
- Python
- Pandas
- Scikit-learn
- Streamlit
""")

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #B0B0B0;
        margin-bottom: 2rem;
    }
    .recommend-card {
        padding: 1rem;
        border-radius: 12px;
        background-color: #111827;
        margin-bottom: 1rem;
        border: 1px solid #2A2A2A;
    }
    .rec-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .rec-meta {
        font-size: 0.95rem;
        color: #D1D5DB;
        margin-bottom: 0.4rem;
    }
    .rec-desc {
        font-size: 0.95rem;
        color: #9CA3AF;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🎬 Netflix Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Find similar Netflix titles using content-based filtering with TF-IDF and cosine similarity.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Metrics
# -----------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Total Titles", len(df))
m2.metric("Movies", len(df[df['type'] == "Movie"]))
m3.metric("TV Shows", len(df[df['type'] == "TV Show"]))

# -----------------------------
# Controls
# -----------------------------
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    selected_title = st.selectbox(
        "Choose a Netflix title",
        options=sorted(df['title'].unique())
    )

with col2:
    selected_type = st.selectbox(
        "Filter by type",
        options=["All", "Movie", "TV Show"]
    )

with col3:
    num_results = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

# -----------------------------
# Selected title details
# -----------------------------
selected_row = df[df['title'] == selected_title].iloc[0]

with st.expander("Selected title details", expanded=True):
    st.write(f"**Title:** {selected_row['title']}")
    st.write(f"**Type:** {selected_row['type']}")
    st.write(f"**Genres:** {selected_row['listed_in']}")
    st.write(f"**Description:** {selected_row['description']}")

# -----------------------------
# Recommendations
# -----------------------------
if st.button("Get Recommendations", use_container_width=True):
    recommendations = recommend(
        selected_title,
        num_recommendations=num_results,
        content_type=selected_type
    )

    if recommendations.empty:
        st.info("Try another title or remove filters for more results.")
    else:
        st.subheader("Recommended Titles")

        for _, row in recommendations.iterrows():
            st.markdown(f"""
                <div class="recommend-card">
                    <div class="rec-title">🎬 {row['title']}</div>
                    <div class="rec-meta"><strong>Type:</strong> {row['type']} | <strong>Genres:</strong> {row['listed_in']}</div>
                    <div class="rec-desc">{row['description']}</div>
                </div>
            """, unsafe_allow_html=True)