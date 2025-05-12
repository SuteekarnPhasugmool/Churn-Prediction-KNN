# Customer Churn Prediction using K-Nearest Neighbors (KNN)

This project focuses on predicting customer churn using the K-Nearest Neighbors (KNN) algorithm on the **Telco Customer Churn dataset**. The objective is to build a machine learning model that accurately identifies customers likely to leave the service, enabling proactive retention strategies.

## 📊 Dataset

- **Source**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: 7,043 customer records
- **Features**: Demographics, service usage, account info, contract type, etc.
- **Target Variable**: `Churn` (Yes/No)

## 🧠 Objectives

- Perform data cleaning and preprocessing.
- Encode categorical variables appropriately.
- Split data into train and test sets.
- Use feature scaling to optimize model performance.
- Apply `GridSearchCV` to tune the number of neighbors (k).
- Evaluate the KNN classifier with multiple metrics.
- Visualize the effect of k on f1-score.

## 🛠 Tools & Libraries

- Python (Jupyter Notebook)
- `pandas`: data manipulation
- `matplotlib`: visualization
- `scikit-learn`: preprocessing, modeling, evaluation

## ✅ Key Steps

1. **Data Preprocessing**
   - Dropped the customerID column.
   - Changed `TotalCharges` column from `object` to `float` and handled missing values.
   - Converted target labels to binary (1 for churn, 0 for no churn).
   - Used `pd.get_dummies()` to encode categorical features.
   - Applied `StandardScaler` after train-test split to normalize the data.

2. **Model Training**
   - Stratified train-test split (80/20) to preserve churn distribution.
   - Applied `GridSearchCV` with 5-fold cross-validation to find the optimal `k`.

3. **Model Evaluation**
   - Selected the best-performing `k` based on cross-validation f1-score.
   - Evaluated final model using:
     - Confusion Matrix
     - Classification Report
     - Accuracy Score

4. **Visualization**
   - Plotted mean cross-validation f1-scores for k values from 1 to 20.

## 📈 Results

- **Best f1-score (CV)**: *0.58*
- **Optimal k**: *19*
- Model demonstrates moderate performance in identifying churners with KNN and proper preprocessing.

## 🚀 Future Improvements

- Compare performance with Logistic Regression, Decision Tree, or Random Forest.
- Handle class imbalance using SMOTE or class weighting.
- Add ROC and Precision-Recall curves.
- Deploy the model with Flask or Streamlit.

## 🔗 References

- Dataset: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- scikit-learn documentation: https://scikit-learn.org/



