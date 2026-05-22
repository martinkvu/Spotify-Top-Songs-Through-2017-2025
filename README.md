# Spotify-Top-Songs-Through-2017-2025


A personal data science project that analyzes Spotify Top Songs data across multiple Wrapped years using machine learning models to uncover listening patterns, mood trends, and artist loyalty over time.

Overview
This pipeline processes Spotify Exportify CSV files (one per Wrapped year) and runs a full suite of analyses — from audio feature engineering to predictive modeling — culminating in 8 publication-quality visualizations.

Features
The project covers data loading and cleaning across all Wrapped years, mood labeling using energy and valence thresholds, PCA-based music taste evolution tracking, Ridge Regression to predict song popularity from audio features, an SVM classifier to predict mood category, a Decision Tree to classify songs by rank tier, a 2026 audio profile forecast using linear extrapolation, and in-depth artist analysis including consistency across years.

Models Used
Ridge Regression predicts song popularity from audio features like danceability, energy, and tempo. Performance is evaluated with R², MAE, and 5-fold cross-validation.
Support Vector Machine (RBF kernel) classifies songs into four mood categories — Happy/Energetic, Angry/Intense, Chill/Positive, and Sad/Calm — derived from energy and valence scores.
Decision Tree (depth=4) classifies songs into rank tiers (Top 10, Mid 11–30, Lower 31+) and surfaces the most important audio features driving rank placement.

Outputs
All results are saved to processed_wrapped/ and include per-year cleaned CSVs, a master dataset, PCA taste vectors, regression coefficients, decision tree rules, mood distributions, artist presence matrices, and predicted 2026 audio profiles.
Visualizations include an overview dashboard, mood heatmap, decision tree diagram, SVM confusion matrix, feature importance chart, and artist presence heatmaps.

Requirements
Python 3.8+, pandas, numpy, scikit-learn, matplotlib.
Install with: pip install pandas numpy scikit-learn matplotlib

Usage
Place your Exportify CSVs named Your_Top_Songs_YYYY.csv in the project directory, update BASE_DIR in the script to match your path, then run python spotify_wrapped_analysis.py. Outputs will appear in the processed_wrapped/ folder.

Project Structure
The base directory holds the raw CSVs and main script. The processed_wrapped/ subfolder contains all generated CSVs, text files, and PNG figures.

Author
Built as a personal portfolio project exploring the intersection of music taste and machine learning.
