Here’s the updated **README** for your project with more advanced details about the evaluation metrics, including your threshold-based approach for **Precision@K** and **Recall@K**.

---

# Music Recommendation System - Final Project

This repository contains the code and documentation for a **Music Recommendation System** developed as part of a machine learning and deep learning course. The system is designed to recommend the top 10 songs to users based on their listening history and preferences using various recommendation techniques such as content-based filtering, collaborative filtering, and deep learning models.

## Project Overview

Music recommendation systems are essential for users to discover new songs they might like. This project utilizes data science and machine learning techniques to create personalized recommendations, enhancing the music discovery experience for users.

### Objective

The primary goal is to develop a recommendation system that suggests songs a user is likely to enjoy, leveraging both content-based and collaborative filtering techniques. We tested and evaluated several models to determine the most effective approach for providing personalized song recommendations.

## Dataset

The project uses the **Taste Profile Subset** from the **Million Song Dataset**. The dataset consists of two main files:

- **Song Data**: Contains metadata about the songs, including `song_id`, `title`, `artist`, `release year`, etc.
- **User Interaction Data**: Contains `user_id`, `song_id`, and `listen_count` (how many times a user played a particular song).

These files were downloaded from external sources, and preprocessing steps were applied to clean and merge the data for use in the recommendation system.

## Models and Approaches

### 1. **Content-Based Filtering**
- **TF-IDF (Term Frequency-Inverse Document Frequency)** was used to vectorize song features such as song title and artist.
- **Cosine Similarity** was then used to calculate the similarity between songs, allowing us to recommend songs that are similar to those a user has already listened to.

### 2. **Collaborative Filtering**
- **Matrix Factorization (SVD)**: A collaborative filtering method that learns latent factors from the interaction matrix (user-song play counts).
- **KNN-Based Collaborative Filtering**: Utilizes the K-Nearest Neighbors algorithm to recommend songs by finding users with similar listening histories.
- **Clustering-Based Collaborative Filtering**: Uses **CoClustering** to group users and songs into clusters, and recommendations are made based on cluster similarities.

### 3. **Deep Learning (Neural Networks)**
- A **Keras-based neural network** was implemented, where user and song embeddings were learned in a latent space. The network predicts a user’s preference for a song based on learned features.

## Evaluation

The models were evaluated using the following advanced metrics, incorporating thresholding to refine recommendation relevance:

### **Threshold-Based Evaluation Approach**
- In this project, we used a **threshold-based system** to define relevance. A song is considered **relevant** if its rating meets or exceeds a certain threshold (e.g., a rating of 3 or more).
- Similarly, a song is considered **recommended** if its predicted rating meets or exceeds a specific threshold (e.g., a prediction score of 3 or more).

This thresholding system allows for more granular control in evaluating the quality of recommendations, ensuring that both predicted and actual ratings are high enough to be considered relevant.

### **Precision@K**
**Precision@K** measures how many of the top K recommendations are relevant, with the additional condition that both the predicted and actual ratings must meet the threshold.

\[
\text{Precision@K} = \frac{\text{# of recommended items @K that are relevant}}{\text{# of recommended items @K}}
\]

For example, if 7 out of the top 10 recommended songs meet the relevance threshold, the **Precision@K** would be:

\[
\text{Precision@10} = \frac{7}{10} = 0.7
\]

### **Recall@K**
**Recall@K** measures how many of the relevant songs (based on actual ratings) were successfully recommended, also considering the prediction threshold.

\[
\text{Recall@K} = \frac{\text{# of recommended items @K that are relevant}}{\text{total # of relevant items}}
\]

For example, if 8 songs in total meet the relevance threshold, and 6 of them are successfully recommended in the top 10 predictions, the **Recall@K** would be:

\[
\text{Recall@10} = \frac{6}{8} = 0.75
\]

### **F1-Score@K**
**F1-Score@K** provides a balance between Precision and Recall, offering a single metric that summarizes the accuracy of the recommendations:

\[
\text{F1\_score@K} = \frac{2 \times \left( \text{Precision@K} \times \text{Recall@K} \right)}{\text{Precision@K} + \text{Recall@K}}
\]

This helps in balancing the trade-off between precision and recall, where one might be higher at the expense of the other.

### **Mean Squared Error (MSE)**
MSE measures the average of the squared differences between actual and predicted values for the play counts. A lower MSE indicates better model performance.

### **Mean Absolute Error (MAE)**
MAE measures the average of the absolute differences between predicted and actual values. It evaluates the accuracy of the predicted play counts and serves as a complement to MSE.

### **Model Comparison**
- **Precision@K** and **Recall@K** were used to compare how well the models recommended relevant songs to users, incorporating the threshold-based relevance.
- **Matrix Factorization (SVD)** emerged as the best-performing model for personalized recommendations.
- **Content-based filtering** (TF-IDF + Cosine Similarity) was useful for finding similar songs but did not perform as well for personalized recommendations as collaborative filtering models.

## Results

- **Matrix Factorization (SVD)** emerged as the best-performing model for personalized song recommendations.
- **Content-based filtering** was effective in recommending similar songs but did not personalize recommendations as well as collaborative filtering techniques.
- A **hybrid approach** combining content-based and collaborative filtering models is recommended for achieving the most accurate and personalized recommendations.

## Installation and Setup

To run the project locally just click on the 'Open in Colab' at the top of the file.

## Usage

1. **Preprocessing**: The notebook walks through data preprocessing, including downloading the dataset, cleaning it, and merging user interactions with song metadata.
2. **Model Building**: Several models are built for generating song recommendations, including content-based filtering, collaborative filtering, and deep learning methods.
3. **Evaluation**: The models are evaluated, and predictions are generated for top 10 song recommendations for each user.

## Future Work

- Implement additional hybrid models that combine content-based and collaborative filtering techniques for improved recommendation accuracy.
- Explore other deep learning architectures, such as recurrent neural networks (RNNs), for capturing sequential patterns in user behavior.
- Experiment with more advanced techniques like **autoencoders** and **reinforcement learning** to improve user experience in the recommendation system.

## Contributors

- **Ariel Hedvat**
- **Shiraz Israeli**
- **Yuval Bakirov**
- **Eitan Bakirov**
