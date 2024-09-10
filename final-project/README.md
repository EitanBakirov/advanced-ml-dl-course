# Music Recommendation System - Final Project

This repository contains the code and documentation for a **Music Recommendation System** project developed as part of a machine learning and deep learning course. The project explores multiple models for recommending songs based on user preferences and item features. The goal is to recommend the top 10 songs to users based on their listening history and other relevant data.

## Group Members
- **Ariel Hedvat**
- **Eitan Bakirov**
- **Yuval Bakirov**
- **Shiraz Israeli**

## Background

A recommendation system uses data to understand user preferences and behavior, suggesting songs and artists that match individual tastes. Our project focuses on building a music recommendation system that enhances music discovery, saving users time and effort while introducing them to new music they’re likely to enjoy.

### Project Goal

The aim of this project was to build a system that suggests the top 10 songs to a user. We experimented with several models, evaluated their performance, and compared them to create an effective recommendation system.

---

## Data

We used the **Million Song Dataset**, which consists of two primary files:
- **Song Data**: This file includes song IDs, titles, release information, artist names, and release years.
- **User Interaction Data**: This file contains user IDs, song IDs, and the corresponding play counts (how many times each user played a particular song).

The data was merged based on the user-song identification key, creating a combined dataset that we used for **Exploratory Data Analysis (EDA)**, preprocessing, and model training.

### Data Preprocessing

1. **Listen Counts to Ratings**: To normalize the user interactions, we converted raw listen counts into ratings using a calculated threshold based on the mean and standard deviation of each user’s listen counts.
   
2. **Dealing with Year Information**: We handled missing and incorrect values in the `year` field, and cleaned the data by reducing the number of users and songs based on interaction thresholds (songs listened to fewer than 200 times or users who listened to fewer than 40 songs were removed).

---

## Models and Approaches

We explored several recommendation models:

### 1. **Rank-Based (Popularity) Model**
This model recommends songs based on overall popularity, ranking songs by their average rating. Unlike other models, it doesn’t take individual preferences into account and doesn’t require train-test splitting or parameter tuning. It is a simple but effective baseline model.

### 2. **Collaborative Filtering (CF) Models**

**Collaborative Filtering** recommends songs by analyzing user preferences for similarities. We used two approaches:
   
#### User-User Similarity-Based Model
This model compares a user's preferences with those of similar users to recommend songs that other similar users liked. We used **KNNBasic** from the Surprise library to compute similarities between users.

Results for the User-User model:
```
MAE: 0.5753, RMSE: 0.7806, Precision@10: 0.922, Recall@10: 0.794, F1 Score: 0.853
```

#### Item-Item Similarity-Based Model
This model recommends songs based on their similarity to songs that a user has already liked. **KNNBasic** from the Surprise library was used to calculate item-item similarities.

Results for the Item-Item model:
```
MAE: 0.5001, RMSE: 0.7309, Precision@10: 0.923, Recall@10: 0.782, F1 Score: 0.847
```

### 3. **Matrix Factorization (SVD)**

**Matrix Factorization (MF)** decomposes the user-item interaction matrix into latent factors, mapping users and songs into a lower-dimensional space. This approach helps predict ratings for new items and make personalized recommendations.

Results for the **SVD** model:
```
MAE: 0.5369, RMSE: 0.7444, Precision@10: 0.915, Recall@10: 0.784, F1 Score: 0.844
```

We also explored using **embedding layers** to enhance matrix factorization. In this approach, user and song IDs were mapped to dense vectors, which were then used to calculate similarity.

Results for the **MF Embedding** model:
```
MAE: 0.5312, RMSE: 0.7288
```

### 4. **Cluster-Based Model**

We applied a **coClustering** model to group users and items into clusters, recommending items based on these clusters. This approach allowed us to uncover hidden patterns in the user-item interaction matrix.

Results for the **coClustering** model:
```
MAE: 0.5505, RMSE: 0.7564, Precision@10: 0.917, Recall@10: 0.778, F1 Score: 0.842
```

### 5. **Content-Based Model**

In the **Content-Based** model, song metadata such as title, artist, and album are used to recommend similar songs. We used **TF-IDF** to weight features and **Cosine Similarity** to find the most similar songs. This model does not rely on user interaction data but rather focuses on the textual features of songs.

Unfortunately, due to runtime issues, we couldn’t fully evaluate the model’s effectiveness.

---

## Model Workflow

For all models (except the Rank-Based model), we followed the same workflow:
1. **Model Explanation**: We explain how the model works and the algorithm behind it.
2. **Hyperparameter Tuning**: We used **grid search cross-validation** from the Surprise library to determine the optimal hyperparameters for each model.
3. **Model Training**: Models were trained using the train set (80% of the data) and evaluated using the test set (20%).
4. **Evaluation Metrics**: We evaluated the models using **Precision@10**, **Recall@10**, **MAE**, and **RMSE**.
5. **User-Specific Examples**: We randomly selected a user from the dataset and compared the recommendations produced by each model.

---

## Evaluation Metrics

### Threshold-Based Evaluation Approach

In this project, we used a **threshold-based system** to define relevance. A song is considered **relevant** if its actual rating meets or exceeds a certain threshold (e.g., a rating of 3 or more). Similarly, a song is considered **recommended** if its predicted rating meets or exceeds a specific threshold (e.g., a prediction score of 3 or more).

This thresholding system ensures that only songs with sufficiently high predicted and actual ratings are considered when calculating evaluation metrics.

### Precision@K

**Precision@K** measures how many of the top K recommended songs are relevant (i.e., meet the relevance threshold for both prediction and rating). It indicates the accuracy of the recommendations at K.

```
Precision@K = (Number of relevant items in top K recommendations) / (Number of items recommended at K)
```

### Recall@K

**Recall@K** measures how many of the relevant songs (based on actual ratings) were successfully recommended in the top K recommendations. This metric focuses on the system's ability to retrieve all relevant items.

```
Recall@K = (Number of relevant items in top K recommendations) / (Total number of relevant items)
```

### F1-Score@K

The **F1-Score@K** balances Precision and Recall. It is useful when combining both metrics to achieve a good balance between recommending enough relevant items and keeping accuracy high.

```
F1-Score@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
```

### Root Mean Squared Error (RMSE)

RMSE measures the average root squared difference between actual and predicted values for play counts. A lower RMSE indicates better performance by the model in predicting user interactions with songs.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values. It helps evaluate how close the predicted play counts are to the actual values, offering a more interpretable error metric alongside MSE.

---

## Results

### Key Model Results
- **User-User Model**: Precision@10 = 0.922, Recall@10 = 0.794, F1 Score = 0.853
- **Item-Item Model**: Precision@10 = 0.923, Recall@10 = 0.782, F1 Score = 0.847
- **Matrix Factorization (SVD)**: Precision@10 = 0.915, Recall@10 = 0.784, F1 Score = 0.844
- **MF Embedding**: MAE = 0.5312, RMSE = 0.7288
- **coClustering Model**: Precision@10 = 0.917, Recall@10 = 0.778, F1 Score = 0.842

---

## Conclusions

Overall, **Matrix Factorization (SVD)** and **Item-Item Similarity** emerged as strong models for making personalized song recommendations. Although the models performed similarly, **Item-Item Similarity** had the lowest RMSE (0.7309), while the **MF Embedding** approach also demonstrated promising results with an RMSE of 0.7288.

We believe that incorporating additional song features (such as genre, mood, and energy level) could further improve the performance
