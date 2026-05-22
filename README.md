# Spotify-Top-Songs-Through-2017-2025


# 🎧 Spotify Wrapped Analysis (2018–2024)

> A personal data science project analyzing Spotify Top Songs across multiple Wrapped years using machine learning to uncover listening patterns, mood trends, and artist loyalty over time.

---

## 📌 Overview

This pipeline processes **Spotify Exportify CSV files** — one per Wrapped year — and runs a full suite of analyses, from audio feature engineering to predictive modeling. It concludes with **8 publication-quality visualizations** saved locally for portfolio use.

---

## ✨ Features

- 🗂️ **Multi-year data loading & cleaning** across all Wrapped years
- 🎭 **Mood labeling** using Energy × Valence quadrants
- 📐 **PCA taste evolution** tracking musical identity shifts over time
- 📈 **Ridge Regression** to predict song popularity from audio features
- 🤖 **SVM Classifier** to predict mood category (4 classes)
- 🌳 **Decision Tree** to classify songs into rank tiers
- 🔮 **2026 Audio Profile Forecast** via linear extrapolation
- 🎤 **Artist Analysis** including consistency scoring across years

---

## 🧠 Models Used

### Ridge Regression
Predicts song **popularity** from audio features (danceability, energy, tempo, etc.).  
Evaluated with R², MAE, and 5-fold cross-validation.

### Support Vector Machine (RBF Kernel)
Classifies songs into **4 mood categories** derived from Energy and Valence:

- 😄 Happy / Energetic
- 😤 Angry / Intense
- 😌 Chill / Positive
- 😔 Sad / Calm

### Decision Tree (depth = 4)
Classifies songs into **rank tiers** — Top 10, Mid (11–30), Lower (31+) — and surfaces the audio features most responsible for rank placement.

---

## 📊 Visualizations

| # | Figure |
|---|--------|
| 1 | Overview Dashboard (PCA, Mood Space, Ridge Coefficients, Popularity Trend) |
| 2 | Mood Distribution Heatmap by Year |
| 3 | Decision Tree Diagram |
| 4 | SVM Confusion Matrix |
| 5 | Decision Tree Feature Importances |
| 6 | Top 15 Artists Overall |
| 7 | Top 5 Artists Per Wrapped Year |
| 8 | Artist Presence Heatmap (Top 15 × Years) |

---

## 📁 Project Structure

```
SUM25Pythons/
├── Your_Top_Songs_2018.csv
├── Your_Top_Songs_2019.csv
│   ...
├── Your_Top_Songs_2024.csv
├── spotify_wrapped_analysis.py
└── processed_wrapped/
    ├── spotify_wrapped_master.csv
    ├── taste_vector_evolution_pca.csv
    ├── mood_distribution_by_year.csv
    ├── predicted_2026_audio_profile.csv
    ├── top_artists_overall.csv
    ├── decision_tree_rules.txt
    ├── fig1_overview_dashboard.png
    │   ...
    └── fig8_artist_presence_heatmap.png
```

---

## ⚙️ Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 🚀 Usage

1. Export your Spotify Wrapped playlists using [Exportify](https://exportify.net/)
2. Name each file `Your_Top_Songs_YYYY.csv` and place them in your project folder
3. Update `BASE_DIR` in the script to match your local path
4. Run the script:

```bash
python spotify_wrapped_analysis.py
```

5. All outputs will be saved to the `processed_wrapped/` folder

---

## 📝 Notes

- Audio features are sourced directly from Spotify's API via Exportify
- Mood labels are rule-based (Energy/Valence thresholds) and validated against the SVM classifier
- The 2026 prediction is a linear trend extrapolation — directional, not prescriptive

---

## 👤 Author

Built as a personal portfolio project exploring the intersection of **music taste** and **machine learning**.
