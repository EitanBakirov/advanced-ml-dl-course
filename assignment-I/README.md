# Bicycle Rental Demand Prediction

This project focuses on predicting daily bicycle rental demand using various machine learning models. The dataset includes features such as weather conditions, temperature, season, and time of day, allowing for the creation of predictive models aimed at forecasting bicycle usage.

## Project Structure

The project is structured as follows:

- `train.csv` and `test.csv`: Contain the dataset used for training and prediction.
- `assignment-I-group-4.ipynb`: The main Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `exercise1.csv`: This csv stores the model results and evaluation metrics.

## Dataset

The dataset includes the following features:

- **Weather conditions** (e.g., temperature, pollution, windspeed)
- **Time-based features** (e.g., hour, day of the week, month)
- **Categorical features** (e.g., season, working day, holiday)

## Models Used

Several machine learning models were used in this project to predict bicycle rental demand:

1. **Random Forest**
2. **Linear Regression**
3. **XGBoost**

Each model was tuned using hyperparameter optimization techniques and evaluated based on the Root Mean Squared Error (RMSE) metric.

## Key Steps

1. **Data Preprocessing**:
   - Handled missing data.
   - Created new features like `log_pollution`, `dayInWeek`, and more.
   - One-hot encoded categorical variables.

2. **Model Training**:
   - Models were trained using cross-validation.
   - Hyperparameters for models such as XGBoost and Random Forest were fine-tuned.

3. **Evaluation**:
   - RMSE was used as the evaluation metric.
   - Model performance and runtime were compared in a summary table.
   - Features contributing the most to predictions were identified for explainability.

## Results

The models were evaluated based on RMSE, with a final comparison across models. The most important features for each model were also identified, highlighting their influence on the predictions.

### Final Scores (RMSE):
- **Random Forest**: 46.75
- **Linear Regression**: 115.98
- **XGBoost**: 40.74

### Top Features:
- **Random Forest**: `hour, log_pollution, temp`
- **Linear Regression**: `PCA (Not Explainable)`
- **XGBoost**: `hour, year, workingday`

### Hyperparameters Used:
- **Random Forest**: `max_features:auto`, `max_depth:30`, `min_samples_leaf:1`
- **Linear Regression**: None
- **XGBoost**: `max_features:auto`, `max_depth:5`

### Features Dropped:
- **Random Forest**: `atemp`, `sunlight`, `traffic`, `pollution`, `holiday`
- **Linear Regression**: `atemp`, `sunlight`, `traffic`, `pollution`, `holiday`
- **XGBoost**: `atemp`, `sunlight`, `traffic`, `pollution`, `windspeed`

### Runtime (Training + Inference):
- **Random Forest**: 1342.78 seconds
- **Linear Regression**: 0.07 seconds
- **XGBoost**: 19.80 seconds

### Hardware Used:
- All models were trained on **CPU**.

## How to Run

At the top of the notebook - "Open in Colab".
After running the notebook, the results will be saved in the `exercise1.csv` file.

## Conclusion

This project demonstrates the application of various machine learning models for demand prediction. Each model was fine-tuned to achieve the lowest RMSE, and the results highlight the most important features influencing the predictions.

