# 🎬 Netflix Recommendation System

## Project Overview
This project is a content-based recommendation system built using the Netflix Titles dataset.

The goal is to recommend similar Netflix titles based on metadata such as:
- genre
- description
- director
- cast
- country

The system uses **TF-IDF vectorization** and **cosine similarity** to identify related content and is deployed as an interactive **Streamlit app**.

---

## Features
- Recommends similar Netflix titles
- Uses content-based filtering
- Interactive Streamlit interface
- Filter by content type (Movie / TV Show)
- Adjustable number of recommendations

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit

---

## Machine Learning Approach
The recommendation engine is based on:

1. **Feature engineering**  
   Relevant text features such as genre, description, cast, director, and country are combined into one text field.

2. **TF-IDF Vectorization**  
   The combined text is converted into numerical vectors.

3. **Cosine Similarity**  
   Similarity scores are calculated between titles to recommend the most relevant content.

---

## Project Files
- `app.py` → Streamlit application
- `recommender.ipynb` → notebook for experimentation and testing
- `netflix_titles.csv` → dataset

---

## How to Run Locally
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn streamlit
