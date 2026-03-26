# Spotify Top Songs Through The Years
# 2018–2024 | Exportify CSV | Portfolio-Grade
# This project Analyzing Top Songs on Personal Spotify from 2017 to 2025 by utilizing support vector model to improve model selection on datasets
# Thus project refine a decision tree model to enhance predictive accuracy on top songs datasets.

# This imports tools needed for this project.
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# This code create the Path Configuration
BASE_DIR = "/Users/marti/Desktop/SUM25Pythons"
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_wrapped")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This code load Spotify Top Songs CSV Files 
csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "Your_Top_Songs_*.csv")))

all_dfs = []

for file in csv_files:
    year = int(os.path.basename(file).split("_")[-1].split(".")[0])
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "Duration (ms)", "Popularity", "Danceability", "Energy",
        "Key", "Loudness", "Mode", "Speechiness", "Acousticness",
        "Instrumentalness", "Liveness", "Valence", "Tempo", "Time Signature"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rank"] = range(1, len(df) + 1)
    df["wrapped_year"] = year
    df["song_id"] = (
        df["Track Name"].str.strip() + " — " + df["Artist Name(s)"].str.strip()
    )
    df["duration_min"] = df["Duration (ms)"] / 60000
    df["is_explicit"] = df["Explicit"].astype(bool)
    df["release_year"] = pd.to_datetime(df["Release Date"], errors="coerce").dt.year

    # This code create mood label based on energy and valence.
    df["mood"] = np.select(
        [
            (df["Energy"] >= 0.5) & (df["Valence"] >= 0.5),
            (df["Energy"] >= 0.5) & (df["Valence"] < 0.5),
            (df["Energy"] < 0.5) & (df["Valence"] >= 0.5),
            (df["Energy"] < 0.5) & (df["Valence"] < 0.5),
        ],
        ["Happy / Energetic", "Angry / Intense", "Chill / Positive", "Sad / Calm"],
        default="Unknown"
    )

    # This code create rank tier for Decision Tree classification.
    df["rank_tier"] = pd.cut(
        df["rank"],
        bins=[0, 10, 30, 999],
        labels=["Top 10", "Mid (11-30)", "Lower (31+)"]
    )

    df.to_csv(os.path.join(OUTPUT_DIR, f"spotify_wrapped_{year}.csv"), index=False)
    all_dfs.append(df)
    print(f"Loaded {year}: {len(df)} tracks")

# This code create a master dataset.

master_df = pd.concat(all_dfs, ignore_index=True)
master_df.to_csv(os.path.join(OUTPUT_DIR, "spotify_wrapped_master.csv"), index=False)
print(f"\nMaster dataset: {len(master_df)} rows | {master_df['wrapped_year'].nunique()} years\n")

# This code analyze song repeat, ranking, and how frequent the artist appear. 

song_repeat = (
    master_df.groupby("song_id")["wrapped_year"].nunique().sort_values(ascending=False)
)
song_repeat.to_csv(os.path.join(OUTPUT_DIR, "song_repeat.csv"))

artist_consistency = (
    master_df.groupby("Artist Name(s)")["wrapped_year"].nunique().sort_values(ascending=False)
)
artist_consistency.to_csv(os.path.join(OUTPUT_DIR, "artist_consistency.csv"))

avg_rank = master_df.groupby("song_id")["rank"].mean().sort_values()
avg_rank.to_csv(os.path.join(OUTPUT_DIR, "song_average_rank.csv"))

# This code does PCA to see the music taste vector evolution. 

pca_features = [
    "Danceability", "Energy", "Loudness",
    "Speechiness", "Acousticness", "Instrumentalness", "Valence", "Tempo"
]

pca_df = master_df[pca_features + ["wrapped_year", "mood"]].dropna().copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_df[pca_features])

pca = PCA(n_components=3)
pca_scores = pca.fit_transform(X_scaled)
pca_df["PC1"] = pca_scores[:, 0]
pca_df["PC2"] = pca_scores[:, 1]
pca_df["PC3"] = pca_scores[:, 2]

taste_evolution = pca_df.groupby("wrapped_year")[["PC1", "PC2", "PC3"]].mean()
taste_evolution.to_csv(os.path.join(OUTPUT_DIR, "taste_vector_evolution_pca.csv"))

print("PCA Explained Variance (PC1–PC3):", np.round(pca.explained_variance_ratio_, 3))
print(f"  Coverage: {pca.explained_variance_ratio_[:3].sum()*100:.1f}%\n")

# This code create a Personal Music DNA.

music_dna = master_df[pca_features].mean()
music_dna.to_csv(os.path.join(OUTPUT_DIR, "personal_music_dna.csv"))

# This code create a Ridge Regression to Predict Popularity based on audio features and music rank.
print("=" * 55)
print("RIDGE REGRESSION — Predicting Song Popularity")

regression_features = [
    "Danceability", "Energy", "Loudness", "Speechiness",
    "Acousticness", "Instrumentalness", "Valence", "Tempo"
]

reg_df = master_df[regression_features + ["Popularity"]].dropna()
X_reg = reg_df[regression_features]
y_reg = reg_df["Popularity"]

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
y_pred_ridge = ridge.predict(X_te)

cv_ridge = cross_val_score(ridge, X_reg, y_reg, cv=5, scoring="r2")

coef_series = pd.Series(ridge.coef_, index=regression_features).sort_values()
coef_series.to_csv(os.path.join(OUTPUT_DIR, "popularity_regression_coefficients.csv"))

print(f"  R²:              {r2_score(y_te, y_pred_ridge):.3f}")
print(f"  MAE:             {mean_absolute_error(y_te, y_pred_ridge):.2f}")
print(f"  CV R² (5-fold):  {cv_ridge.mean():.3f} ± {cv_ridge.std():.3f}")

# This does the Support Vector Machine to classify the Mood Type in the audio features.
print("SVM CLASSIFIER — Mood Prediction (4 classes)")

mood_features = [
    "Danceability", "Energy", "Loudness",
    "Speechiness", "Acousticness", "Valence", "Tempo"
]

svm_df = master_df[mood_features + ["mood"]].dropna()
svm_df = svm_df[svm_df["mood"] != "Unknown"]

le = LabelEncoder()
y_mood = le.fit_transform(svm_df["mood"])
X_mood = StandardScaler().fit_transform(svm_df[mood_features])

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_mood, y_mood, test_size=0.25, random_state=42, stratify=y_mood
)

svm_clf = SVC(kernel="rbf", C=1.5, gamma="scale", random_state=42)
svm_clf.fit(X_tr2, y_tr2)
y_pred_svm = svm_clf.predict(X_te2)

cv_svm = cross_val_score(svm_clf, X_mood, y_mood, cv=5, scoring="accuracy")

print(f"  Accuracy:        {accuracy_score(y_te2, y_pred_svm):.3f}")
print(f"  CV Acc (5-fold): {cv_svm.mean():.3f} ± {cv_svm.std():.3f}")
print("\n  Classification Report:")
print(classification_report(y_te2, y_pred_svm, target_names=le.classes_))

mood_dist = (
    master_df[master_df["mood"] != "Unknown"]
    .groupby(["wrapped_year", "mood"])
    .size().unstack(fill_value=0)
)
mood_dist.to_csv(os.path.join(OUTPUT_DIR, "mood_distribution_by_year.csv"))

# This code does a Decision Tree to rank tier classification based on music features. 

print("\n" + "=" * 55)
print("DECISION TREE — Rank Tier Classification")

dt_df = master_df[regression_features + ["rank_tier"]].dropna()
y_tier = dt_df["rank_tier"].astype(str)
X_tier = dt_df[regression_features]

X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
    X_tier, y_tier, test_size=0.25, random_state=42, stratify=y_tier
)

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42
)
dt.fit(X_tr3, y_tr3)
y_pred_dt = dt.predict(X_te3)

cv_dt = cross_val_score(dt, X_tier, y_tier, cv=5, scoring="accuracy")

print(f"  Accuracy:        {accuracy_score(y_te3, y_pred_dt):.3f}")
print(f"  CV Acc (5-fold): {cv_dt.mean():.3f} ± {cv_dt.std():.3f}")
print("\n  Classification Report:")
print(classification_report(y_te3, y_pred_dt))

fi = pd.Series(dt.feature_importances_, index=regression_features).sort_values(ascending=False)
fi.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_feature_importances.csv"))
print("  Top Feature Importances:")
print(fi.round(4))

rules_text = export_text(dt, feature_names=regression_features)
with open(os.path.join(OUTPUT_DIR, "decision_tree_rules.txt"), "w") as f:
    f.write(rules_text)
print("\n  Decision rules → decision_tree_rules.txt")

# This code show what the audio profile will be like based on past audio features of top songs.

yearly_means = master_df.groupby("wrapped_year")[pca_features].mean()
future_prediction = {}
for feature in pca_features:
    lr = LinearRegression()
    lr.fit(yearly_means.index.values.reshape(-1, 1), yearly_means[feature].values)
    future_prediction[feature] = lr.predict([[2026]])[0]

future_profile = pd.Series(future_prediction)
future_profile.to_csv(os.path.join(OUTPUT_DIR, "predicted_2026_audio_profile.csv"))
print("\n" + "=" * 55)
print("PREDICTED 2026 AUDIO PROFILE:")
print(future_profile.round(3))

# This show top songs and average popularity by each year. 

top_songs_each_year = (
    master_df.sort_values(["wrapped_year", "rank"])
    .groupby("wrapped_year").first().reset_index()
    [["wrapped_year", "Track Name", "Artist Name(s)", "Popularity",
      "Energy", "Valence", "Danceability", "rank"]]
    .rename(columns={"Track Name": "top_track", "Artist Name(s)": "artist"})
)
top_songs_each_year.to_csv(os.path.join(OUTPUT_DIR, "top_song_each_year.csv"), index=False)
print("\nTop song of each year:")
print(top_songs_each_year[["wrapped_year", "top_track", "artist"]].to_string(index=False))

avg_popularity_by_year = (
    master_df.groupby("wrapped_year")["Popularity"]
    .mean().round(2).reset_index()
    .rename(columns={"Popularity": "avg_popularity"})
)
avg_popularity_by_year.to_csv(os.path.join(OUTPUT_DIR, "average_popularity_by_year.csv"), index=False)

# This code provide Top Artist analysis overall and by each year based on the number of time their songs appeared in the top songs.

print("\n" + "=" * 55)
print("TOP ARTIST ANALYSIS")

# Build exploded artist table (one row per artist per song)
artist_df = master_df.copy()
artist_df["artists_list"] = artist_df["Artist Name(s)"].str.split(";")
artist_exploded = artist_df.explode("artists_list")
artist_exploded["artist"] = artist_exploded["artists_list"].str.strip()
artist_exploded = artist_exploded[artist_exploded["artist"].notna() & (artist_exploded["artist"] != "")]

# This focus on Overall Artists
overall_artists = (
    artist_exploded
    .groupby("artist")
    .agg(
        track_count    = ("Track Name",    "count"),
        years_appeared = ("wrapped_year",  "nunique"),
        avg_rank       = ("rank",          "mean"),
        best_rank      = ("rank",          "min"),
        avg_popularity = ("Popularity",    "mean"),
    )
    .sort_values(["track_count", "avg_rank"], ascending=[False, True])
    .reset_index()
)
overall_artists.to_csv(os.path.join(OUTPUT_DIR, "top_artists_overall.csv"), index=False)

print("\nTop 15 Artists — Overall (by track appearances):")
print(
    overall_artists.head(15)
    [["artist", "track_count", "years_appeared", "avg_rank", "best_rank", "avg_popularity"]]
    .round(2)
    .to_string(index=False)
)

# This focus on Top Artist of Each Year. 
artists_by_year = (
    artist_exploded
    .groupby(["wrapped_year", "artist"])
    .agg(
        track_count    = ("Track Name",   "count"),
        avg_rank       = ("rank",         "mean"),
        best_rank      = ("rank",         "min"),
        avg_popularity = ("Popularity",   "mean"),
    )
    .reset_index()
    .sort_values(["wrapped_year", "track_count", "avg_rank"], ascending=[True, False, True])
)
artists_by_year.to_csv(os.path.join(OUTPUT_DIR, "top_artists_by_year.csv"), index=False)

print("\nTop 5 Artists Per Year:")
for yr in sorted(artists_by_year["wrapped_year"].unique()):
    top5 = artists_by_year[artists_by_year["wrapped_year"] == yr].head(5)
    entries = [
        f"{row['artist']} ({int(row['track_count'])} tracks, avg rank {row['avg_rank']:.1f})"
        for _, row in top5.iterrows()
    ]
    print(f"\n  {yr}:")
    for e in entries:
        print(f"    • {e}")

# This shows which artists appeared across multiple Wrapped years.
top_n_artists = overall_artists.head(15)["artist"].tolist()

presence_matrix = (
    artist_exploded[artist_exploded["artist"].isin(top_n_artists)]
    .groupby(["artist", "wrapped_year"])["Track Name"]
    .count()
    .unstack(fill_value=0)
    .reindex(top_n_artists)          # keep ranking order
)
presence_matrix.to_csv(os.path.join(OUTPUT_DIR, "artist_presence_matrix.csv"))

# This show which artists appear more consistently for 3 or more years. 
consistent_artists = overall_artists[overall_artists["years_appeared"] >= 3].copy()
consistent_artists.to_csv(os.path.join(OUTPUT_DIR, "top_artists_consistent.csv"), index=False)

print(f"\nArtists appearing in 3+ Wrapped years ({len(consistent_artists)} total):")
print(
    consistent_artists[["artist", "years_appeared", "track_count", "avg_popularity"]]
    .head(10)
    .round(2)
    .to_string(index=False)
)

# This provide the visualizations of this project. 

plt.style.use("seaborn-v0_8-darkgrid")

# --- Figure 1: 2×2 Dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Spotify Wrapped — Audio Dashboard (2018–2024)", fontsize=15, fontweight="bold")

# (a) PCA taste evolution
for col in ["PC1", "PC2", "PC3"]:
    axes[0, 0].plot(taste_evolution.index, taste_evolution[col], marker="o", label=col)
axes[0, 0].set_title("Taste Vector Evolution (PCA)")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Mean PCA Score")
axes[0, 0].legend()

# (b) Energy vs Valence scatter (colored by year)
sc = axes[0, 1].scatter(
    master_df["Energy"], master_df["Valence"],
    c=master_df["wrapped_year"], cmap="viridis", alpha=0.5, s=15
)
plt.colorbar(sc, ax=axes[0, 1], label="Year")
axes[0, 1].axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
axes[0, 1].axvline(0.5, color="gray", linewidth=0.8, linestyle="--")
axes[0, 1].set_title("Mood Space (Energy × Valence)")
axes[0, 1].set_xlabel("Energy")
axes[0, 1].set_ylabel("Valence")

# (c) Ridge coefficients (popularity)
coef_series.plot(kind="barh", ax=axes[1, 0], color="#1DB954")
axes[1, 0].set_title("Ridge: Feature Impact on Popularity")
axes[1, 0].set_xlabel("Coefficient")
axes[1, 0].axvline(0, color="gray", linewidth=0.8)

# (d) Average popularity trend
axes[1, 1].bar(
    avg_popularity_by_year["wrapped_year"],
    avg_popularity_by_year["avg_popularity"],
    color="#1DB954", edgecolor="#191414"
)
axes[1, 1].set_title("Average Song Popularity by Year")
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Avg Popularity (0–100)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_overview_dashboard.png"), dpi=150)
plt.show()

# --- Figure 2: Mood heatmap ---
fig, ax = plt.subplots(figsize=(10, 5))
mood_dist_pct = mood_dist.div(mood_dist.sum(axis=1), axis=0) * 100
im = ax.imshow(mood_dist_pct.T, aspect="auto", cmap="YlGn")
ax.set_xticks(range(len(mood_dist_pct.index)))
ax.set_xticklabels(mood_dist_pct.index)
ax.set_yticks(range(len(mood_dist_pct.columns)))
ax.set_yticklabels(mood_dist_pct.columns)
ax.set_title("Mood Distribution by Year (%) — SVM-Validated Labels")
ax.set_xlabel("Wrapped Year")
plt.colorbar(im, ax=ax, label="% of songs")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_mood_heatmap.png"), dpi=150)
plt.show()

# --- Figure 3: Decision Tree ---
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(
    dt,
    feature_names=regression_features,
    class_names=dt.classes_,
    filled=True, rounded=True, fontsize=8, ax=ax
)
ax.set_title("Decision Tree — Rank Tier Classification (depth=4)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_decision_tree.png"), dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 4: SVM Confusion Matrix ---
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_te2, y_pred_svm),
    display_labels=le.classes_
)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("SVM Mood Classifier — Confusion Matrix")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_svm_confusion_matrix.png"), dpi=150)
plt.show()

# --- Figure 5: Decision Tree Feature Importances ---
fig, ax = plt.subplots(figsize=(8, 5))
fi.sort_values().plot(kind="barh", color="#1DB954", edgecolor="#191414", ax=ax)
ax.set_title("Decision Tree — Feature Importances for Rank Tier")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_dt_feature_importances.png"), dpi=150)
plt.show()

# --- Figure 6: Top 15 Artists Overall (horizontal bar) ---
top15 = overall_artists.head(15).sort_values("track_count")

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    top15["artist"],
    top15["track_count"],
    color="#1DB954", edgecolor="#191414"
)
# Annotate each bar with years appeared
for bar, (_, row) in zip(bars, top15.iterrows()):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{int(row['years_appeared'])}yr{'s' if row['years_appeared'] > 1 else ''}",
        va="center", fontsize=8, color="gray"
    )
ax.set_title("Top 15 Artists Overall — Track Appearances (2018–2024)", fontweight="bold")
ax.set_xlabel("Number of Tracks")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_top_artists_overall.png"), dpi=150)
plt.show()

# --- Figure 7: Top 5 Artists Per Year (grouped bar chart) ---
top5_per_year = (
    artists_by_year
    .groupby("wrapped_year")
    .head(5)
    .copy()
)
top5_per_year["label"] = top5_per_year["artist"].str[:18]   # truncate long names

years = sorted(top5_per_year["wrapped_year"].unique())
n_years = len(years)
fig, axes = plt.subplots(1, n_years, figsize=(16, 6), sharey=False)
fig.suptitle("Top 5 Artists Per Wrapped Year — Track Count", fontsize=13, fontweight="bold")

for ax, yr in zip(axes, years):
    data = top5_per_year[top5_per_year["wrapped_year"] == yr].head(5)
    ax.barh(data["label"], data["track_count"], color="#1DB954", edgecolor="#191414")
    ax.set_title(str(yr), fontweight="bold")
    ax.set_xlabel("Tracks")
    ax.invert_yaxis()   # rank 1 at top

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_top_artists_per_year.png"), dpi=150)
plt.show()

# --- Figure 8: Artist Presence Heatmap (top 15 artists × years) ---
fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(presence_matrix.values, aspect="auto", cmap="YlGn")

ax.set_xticks(range(len(presence_matrix.columns)))
ax.set_xticklabels(presence_matrix.columns, fontsize=10)
ax.set_yticks(range(len(presence_matrix.index)))
ax.set_yticklabels(presence_matrix.index, fontsize=9)

# Annotate cells with track counts
for i in range(len(presence_matrix.index)):
    for j in range(len(presence_matrix.columns)):
        val = presence_matrix.values[i, j]
        if val > 0:
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=9, color="black", fontweight="bold")

plt.colorbar(im, ax=ax, label="Tracks in Wrapped")
ax.set_title("Artist Presence Across Wrapped Years (Top 15 Artists)", fontweight="bold")
ax.set_xlabel("Wrapped Year")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_artist_presence_heatmap.png"), dpi=150)
plt.show()

print("\n✅ PIPELINE COMPLETE — outputs saved to:", OUTPUT_DIR)