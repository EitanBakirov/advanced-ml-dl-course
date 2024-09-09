# TV Show Dialogue Classification: Friends vs. Seinfeld

This project uses machine learning and deep learning techniques to classify dialogues from two popular TV shows, **Friends** and **Seinfeld**. The notebook explores various preprocessing techniques, text representation methods, model comparisons, and result visualization to build an accurate classifier.

## Project Structure

- **Dataset**: Contains dialogues from both Friends and Seinfeld, with labels indicating which show the dialogue belongs to.
- **Notebook**: Processes the text data, trains models, and evaluates their performance across various metrics.
- **Results**: Includes model accuracy, performance comparisons, and word cloud visualizations for each show.

## Key Sections

### 1. Exploratory Data Analysis (EDA)
- **Data Overview**: Inspected for missing values (none found), and the dataset is fairly balanced between Friends and Seinfeld.
- **Dialogue Length & Distribution**: Explored how dialogues are distributed across characters in each show. Jerry from Seinfeld has the most dialogues, while Friends has a more even distribution.
- **Word Frequency**: Common words are explored, and stopwords (e.g., "the", "is") are removed to focus on more meaningful terms.

### 2. General Preprocessing
- **Text Cleaning**: Special characters and stopwords are removed.
- **Stemming**: Words are reduced to their base forms to standardize the text (e.g., "running" becomes "run").

### 3. Text Representation Techniques
- **Bag of Words (BOW)**: Converts the text into a matrix of word counts. Simple but effective, though it loses word order.
- **TF-IDF**: Weights words based on their importance in the dataset. This helps reduce dimensionality and focus on key terms.
- **Word Embeddings**: Capture semantic relationships between words, allowing for more meaningful text representation, particularly in deep learning models.

### 4. Split-Based Preprocessing
- **Labeling**: Dialogues are labeled as binary values (0 for Seinfeld, 1 for Friends).
- **Tokenization & Padding**: Text is tokenized and padded to ensure consistent input length for neural networks.

### 5. Modeling
- Several models are trained on the preprocessed data using different text representation techniques:
  - **Logistic Regression**: Applied to both BOW and TF-IDF representations.
  - **Random Forest**: Also tested with BOW and TF-IDF.
  - **Convolutional Neural Network (CNN)**: Built using word embeddings, with an architecture that includes convolutional layers, dense layers, and dropout for regularization.

### 6. Comparison of Models

- **Logistic Regression (BOW)** and the **Convolutional Neural Network (CNN)** performed the best in terms of accuracy and ROC-AUC scores.
- The CNN was selected as the best model due to its overall performance.

### 7. Results Exploration

1. **Visualization**:
   - The accuracy for each character in both **Friends** and **Seinfeld** is visualized.
   - As expected, characters from **Friends** have higher accuracy, possibly due to a larger number of dialogues available for training. However, the accuracy for both shows is relatively high, indicating good model performance.

2. **Word Clouds**:
   - Word clouds for both shows are generated to visualize the most frequently used words. It’s observed that both shows use similar words most of the time, which makes sense given that they are both American sitcoms.

### 8. Final Results

Based on the best-performing model (CNN), the final results are:

- **Test Set Accuracy**: 83.27%
- **Train Set Accuracy**: 81.20%
- **Number of Trainable Parameters**: 1,161,873
- **Number of Layers**: 7
- **Regularization Method**: Dropout
- **Number of Epochs**: 2
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Embedding Dimension**: 32

## How to Run

At the top of the file - "Open in Colab".<br>
After running the notebook, the results will be saved in the `exercise3.csv` file.

## Conclusion

This project successfully classifies dialogues from Friends and Seinfeld using a combination of machine learning and deep learning models. The CNN model with word embeddings proved to be the most effective, and various preprocessing and text representation techniques were applied to optimize performance.
