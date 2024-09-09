Here's a sample **README** for your music recommendation system project:

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

The models were evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the squared difference between actual and predicted values.
- **Mean Absolute Error (MAE)**: Measures the absolute difference between actual and predicted values.
- **Precision and Recall**: Evaluates the relevance and accuracy of the recommended songs.

## Results

- **Matrix Factorization (SVD)** emerged as the best-performing model for personalized song recommendations.
- **Content-based filtering** was effective in recommending similar songs but did not personalize recommendations as well as collaborative filtering techniques.
- A **hybrid approach** combining content-based and collaborative filtering models is recommended for achieving the most accurate and personalized recommendations.

## Installation and Setup

To run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/recommendation-system-project.git
    cd recommendation-system-project
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook recommendation-system-project.ipynb
    ```

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to update the repository URL, contributor names, and other details as needed. Let me know if you'd like to modify anything in the README!
