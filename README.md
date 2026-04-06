# 🎬 Netflix Recommendation System

## 📌 Project Overview

This project is a **content-based recommendation system** that suggests similar Netflix titles based on metadata such as genre, description, cast, director, and country.

The system uses **TF-IDF vectorization** and **cosine similarity** to compute similarities between titles and provide relevant recommendations.

An interactive web application was built using **Streamlit** to allow users to explore recommendations in a user-friendly interface.

---

## 🚀 Features

* Recommend similar Netflix movies and TV shows
* Content-based filtering using machine learning
* Interactive web interface with Streamlit
* Filter by content type (Movie / TV Show)
* Adjustable number of recommendations
* Clean and intuitive UI

---

## 🧠 How It Works

### 1. Feature Engineering

Relevant text features are combined:

* Genre (`listed_in`)
* Description
* Cast
* Director
* Country

### 2. TF-IDF Vectorization

Text data is converted into numerical vectors using **TF-IDF**.

### 3. Cosine Similarity

Similarity between titles is computed using **cosine similarity** to recommend the most relevant content.

---

## 📂 Project Structure

```
netflix-recommendation-system/
│
├── app.py                 # Streamlit web application
├── netflix_titles.csv     # Dataset
├── recommender.ipynb      # Development notebook
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/SalmaMork/netflix-recommendation-system.git
cd netflix-recommendation-system
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the application

```bash
streamlit run app.py
```

---

### 4. Open in browser

Go to:

```
http://localhost:8501
```

---

## 💡 Example

Input:

```
Breaking Bad
```

Output:

* Better Call Saul
* The Lincoln Lawyer
* Other similar crime/drama content

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit

---

## 📈 Key Skills Demonstrated

* Data cleaning and preprocessing
* Feature engineering
* Natural Language Processing (TF-IDF)
* Similarity modeling (cosine similarity)
* Building interactive applications with Streamlit
* End-to-end project development

---

## 📌 Future Improvements

* Add collaborative filtering
* Improve recommendation accuracy
* Deploy application online
* Add user preferences and ratings

---

## 👩‍💻 Author

**Salma Morkani**
