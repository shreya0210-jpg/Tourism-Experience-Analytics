# 🌍 Tourism Experience Analytics

## 📌 Project Overview

Tourism Experience Analytics is a Machine Learning project that analyzes tourism data and provides:

- 📈 Rating Prediction
- 🔍 Visit Mode Prediction
- ⭐ Attraction Recommendations
- 📊 Tourism Analytics & Visualizations

---

## 📊 Dataset

- **Records:** 52,930
- **Columns:** 12
- **Missing Values:** 0

### Features
`UserId`, `VisitYear`, `VisitMonthNum`, `VisitModeId`, `AttractionId`, `Rating`, `ContinentId`, `RegionId`, `CountryId`, `CityId`, `AttractionCityId`, `AttractionTypeClean`

---

## ⚙️ Feature Engineering

New features created:

- UserVisitCount
- UserAverageRating
- AttractionAverageRating

Final dataset shape: **52,930 × 15**

---

## 🤖 Machine Learning Models

### 📈 Regression – Rating Prediction

| Model | MSE | R² Score |
|---|---:|---:|
| Linear Regression | 0.9031 | 0.0411 |
| Random Forest Regression | 0.8374 | 0.1109 |

🏆 **Best Model: Random Forest Regression**

---

### 🔍 Classification – Visit Mode Prediction

**Model:** Random Forest Classifier

- Accuracy: **52.78%**
- Precision: **52.38%**
- Recall: **52.78%**
- F1 Score: **51.07%**

---

## ⭐ Recommendation System

**Method:** Content-Based Recommendation

- TF-IDF Vectorization
- Cosine Similarity
- Total Attractions: **30**
- Highest Similarity Score: **0.8747**

---

## 📊 Analytics Dashboard

The Streamlit app includes:

- 🏆 Top Attractions by Average Rating
- 🌍 Top Regions
- 🏛️ Attraction Type Distribution
- ⭐ Average Rating by Region
- 🌎 Top Countries

---

## 🌐 Live Application

👉 [Tourism Experience Analytics App](https://tourism-experience-analytics-vwhup7vdrysjulsgrgv9af.streamlit.app/)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- TF-IDF
- Cosine Similarity

---

## 📁 Project Structure

```text
├── app.py
├── best_regression_model.pkl
├── best_classification_model.pkl
├── recommendation_data.pkl
├── similarity_matrix.pkl
├── requirements.txt
└── README.md

🎯 Conclusion

This project successfully performs tourism data analysis, rating prediction, visit mode classification, and attraction recommendations through an interactive Streamlit application.

👩‍💻 Author

Shreya Tripathi
B.Tech – Computer Science Engineering
Specialization: Artificial Intelligence & Machine Learning
