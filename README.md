# Spotify-Music-Track-Popularity-Analysis
An analysis of Spotify track popularity based on audio features and artist size. This is a project for the course dsc80 taken at UCSD.

**Andy Cai**

---

# Introduction

Spotify is one of the world's largest music streaming platforms, hosting millions of tracks across a wide variety of genres. In addition to providing music to listeners, Spotify generates detailed audio features for each song, such as danceability, energy, acousticness, tempo, and valence. These features offer a quantitative description of music and create opportunities to analyze what factors contribute to a song's success.

This project investigates the relationship between artist popularity and track popularity on Spotify. Specifically, it explores whether tracks from artists with larger audiences tend to be more popular and whether artist characteristics are stronger predictors of success than a song's audio features. It also examines which musical attributes are associated with higher popularity and develops a machine learning model to predict whether a song will become popular.

The dataset contains over 100,000 Spotify tracks spanning 114 genres. For this analysis, I focused on five genres: EDM, Rock, Hip-Hop, R&B, and Country. These genres differ substantially in production techniques, instrumentation, and audience demographics, making them useful for comparison.

Relevant columns include:

| Column | Description |
|----------|-------------|
| popularity | Spotify popularity score for a track |
| followers | Number of followers for an artist |
| artist_popularity | Spotify popularity score for an artist |
| danceability | How suitable a track is for dancing |
| energy | Measure of intensity and activity |
| acousticness | Confidence measure of whether a track is acoustic |
| valence | Musical positivity of a track |
| tempo | Estimated beats per minute |
| track_genre | Genre classification of the track |
| release_date | Release date of the track |

The primary question explored in this project is:

**Do tracks from artists with larger follower counts tend to be more popular, and can track popularity be predicted using information available at release time?**

---

# Data Cleaning and Exploratory Data Analysis

## Data Cleaning

Several preprocessing steps were performed before analysis:

- Removed non-informative columns such as `Unnamed: 0`, `track_id`, and `album_name`.
- Created a `primary_artist` column by extracting the first artist listed in tracks with multiple artists.
- Merged track-level and artist-level datasets using artist names.
- Converted `release_date` into datetime format and extracted a `release_year` feature.
- Investigated missing values before deciding how to handle them.
- Restricted analysis to five genres: EDM, Rock, Hip-Hop, R&B, and Country.

After merging the datasets, 4,961 of the 5,000 selected tracks successfully matched an artist record, resulting in less than 1% data loss.

### Head of Cleaned DataFrame

```python
merged.head()
```
print(merged.head().to_markdown(index=False))
print(merged.head().to_markdown(index=False))

## Univariate Analysis

### Distribution of Popularity Scores

<iframe
    src="assets/popularity_distribution1.html"
    width="800"
    height="600"
    frameborder="0">
</iframe>

<iframe
    src="assets/popularity_distribution2.html"
    width="800"
    height="600"
    frameborder="0">
</iframe>

<iframe
    src="assets/popularity_distribution3.html"
    width="800"
    height="600"
    frameborder="0">
</iframe>

<iframe
    src="assets/popularity_distribution4.html"
    width="800"
    height="600"
    frameborder="0">
</iframe>
Track popularity is highly right-skewed, with a substantial spike at zero. Approximately **14.05% of tracks** and **43.09% of artists** have popularity scores equal to zero. This suggests that a large portion of songs and artists receive little measurable engagement on Spotify.

Because popularity is the primary outcome of interest, these zero-valued observations were retained in the dataset.

## Bivariate Analysis

### Track Popularity Across Genres

<iframe
    src="assets/track_pop_genre.html"
    width="800"
    height="600"
    frameborder="0">
</iframe>

Popularity distributions differ across genres. EDM and Hip-Hop tracks tend to exhibit higher median popularity, while Rock and Country contain a larger proportion of low-popularity tracks. These differences suggest that genre may be useful in predicting track popularity.

## Interesting Aggregates

### Average Popularity by Genre

| Genre | Relative Average Popularity |
|--------|---------------------------|
| EDM | Highest |
| Hip-Hop | High |
| R&B | Moderate |
| Rock | Lower |
| Country | Lower |

Grouping tracks by genre revealed meaningful differences in average popularity. Genres with higher average danceability and energy also tended to have higher popularity scores. These aggregate statistics suggest that musical characteristics may help explain differences in popularity across genres.

---

# Assessment of Missingness

The `tempo` column contains a substantial number of missing values and was selected for missingness analysis.

## NMAR Analysis

Tempo is a plausible candidate for **NMAR** missingness. Tempo values are estimated algorithmically from audio signals, and tracks with ambiguous rhythmic structures, spoken-word recordings, or highly experimental music may be more likely to have missing tempo values. In this case, the probability that tempo is missing may depend on the underlying tempo itself.

Additional information about Spotify's tempo-detection process would be required to determine whether the missingness mechanism is truly NMAR.

## Missingness Dependency

### Does Tempo Missingness Depend on Genre?

**Null Hypothesis:** Tempo missingness is independent of track genre.

**Alternative Hypothesis:** Tempo missingness depends on track genre.

A permutation test using Total Variation Distance (TVD) produced:

- Observed TVD = 0.179
- p-value < 0.001

Because the p-value is extremely small, I reject the null hypothesis and conclude that tempo missingness depends on track genre.

<iframe
    src="assets/temp_dist_plot.html"
    width="800"
    height="600"
    frameborder="0">
</iframe>

### Does Tempo Missingness Depend on Musical Key?

**Null Hypothesis:** Tempo missingness is independent of musical key.

**Alternative Hypothesis:** Tempo missingness depends on musical key.

Results:

- Observed TVD = 0.048
- p-value = 0.472

Because the p-value is much larger than 0.05, I fail to reject the null hypothesis and conclude that there is insufficient evidence that tempo missingness depends on musical key.

Overall, the evidence suggests that tempo missingness is most consistent with a **MAR** mechanism because it depends on an observed variable (`track_genre`).

---

# Hypothesis Testing

This section investigates whether artists with larger follower counts tend to have more popular tracks.

### Hypotheses

**Null Hypothesis:** There is no linear relationship between artist follower count and track popularity.

**Alternative Hypothesis:** There is a positive linear relationship between artist follower count and track popularity.

### Test Statistic

Pearson correlation coefficient.

### Significance Level

α = 0.05

### Results

- Observed correlation = 0.080
- p-value < 0.001

### Conclusion

I reject the null hypothesis and conclude that there is evidence of a positive relationship between artist follower count and track popularity.

However, the observed correlation is very weak. While artists with larger audiences tend to have slightly more popular tracks, follower count explains only a small portion of the variation in popularity. Other factors such as genre, release timing, playlist exposure, and musical characteristics likely contribute more strongly to track success.

---

# Framing a Prediction Problem

The goal is to predict whether a track will achieve a popularity score of at least 70.

This is a **binary classification problem**.

### Response Variable

`popular_track`

- 1 = popularity ≥ 70
- 0 = popularity < 70

### Features Available at Prediction Time

- Danceability
- Energy
- Acousticness
- Speechiness
- Instrumentalness
- Valence
- Loudness
- Tempo
- Duration
- Explicit status
- Release year
- Genre

These variables are known when a track is released and therefore can be used for prediction.

### Evaluation Metric

I use the **F1 score** as the evaluation metric.

The target classes are imbalanced:

- Popular tracks: 635
- Non-popular tracks: 4,365

Because accuracy can be misleading under class imbalance, the F1 score provides a better balance between precision and recall.

---

# Baseline Model

The baseline model is a **Logistic Regression classifier**.

### Features Used

- Danceability (quantitative)
- Energy (quantitative)

Both variables were standardized using a `StandardScaler`.

### Model Performance

| Model | F1 Score |
|---------|----------|
| Logistic Regression | 0.229 |

The baseline model demonstrates limited predictive ability. While danceability and energy contain some information about popularity, they are insufficient for accurately identifying highly popular tracks.

---

# Final Model

To improve predictive performance, additional features and feature engineering were introduced.

## Engineered Features

### Log Duration

A logarithmic transformation was applied to track duration:

```python
log_duration = np.log1p(duration_ms)
```

This transformation reduces the influence of extremely long tracks while preserving the ordering of durations.

### Release Decade

Tracks were grouped into decade-level categories using release year. This feature captures broader historical trends in music consumption and production.

## Additional Features

- Acousticness
- Speechiness
- Instrumentalness
- Valence
- Loudness
- Tempo
- Explicit
- Release Year

These variables provide a richer description of a track's musical characteristics.

## Model Choice

I selected a **Random Forest Classifier** because track popularity is likely influenced by nonlinear relationships and interactions among audio features. Random forests can capture these patterns more effectively than logistic regression.

## Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used.

Best hyperparameters:

| Hyperparameter | Value |
|--------------|--------|
| n_estimators | 300 |
| max_depth | 5 |
| min_samples_split | 2 |

### Model Performance

| Model | F1 Score |
|---------|----------|
| Baseline Logistic Regression | 0.229 |
| Final Random Forest | 0.298 |

The final model improved the F1 score by approximately 30%.

The additional audio features and engineered variables provide more information about track characteristics and historical trends, allowing the Random Forest model to better distinguish between popular and non-popular tracks.

---

# Fairness Analysis

To evaluate fairness, I compared model performance between explicit and non-explicit tracks.

### Groups

- Explicit tracks
- Non-explicit tracks

### Evaluation Metric

F1 Score

### Hypotheses

**Null Hypothesis:** The model performs equally well across explicit and non-explicit tracks. Any difference in F1 score is due to random chance.

**Alternative Hypothesis:** The model performs worse on explicit tracks than on non-explicit tracks.

### Results

| Group | F1 Score |
|---------|----------|
| Explicit | 0.404 |
| Non-Explicit | 0.275 |

Observed difference:

```text
F1(non-explicit) − F1(explicit) = -0.129
```

Permutation test results:

- p-value = 0.927

### Conclusion

I fail to reject the null hypothesis.

Although the model performs better on explicit tracks than on non-explicit tracks, the observed difference is consistent with random variation. Based on this analysis, there is insufficient evidence that the model is unfair with respect to explicit content.

---

**Author:** Andy Cai  
**Project:** Spotify Music Track Popularity Analysis  
**Course:** DSC 80
