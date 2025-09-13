# Project: Credit Risk Prediction

This project aims to predict credit risk using machine learning, classifying loan applicants into risk categories based on financial and personal data.

## Table of Contents
- [Project: Credit Risk Prediction](#project-credit-risk-prediction)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
    - [Handling Missing Values](#handling-missing-values)
    - [Handling Imbalanced Data](#handling-imbalanced-data)
    - [Handling Outliers](#handling-outliers)
    - [Encoding Categorical Variables](#encoding-categorical-variables)
    - [Feature Scaling](#feature-scaling)
    - [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
  - [Methodology](#methodology)
    - [Models Used](#models-used)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Evaluation](#evaluation)
  - [Results](#results)
  - [Conclusion](#conclusion)

## Introduction
Credit risk prediction helps financial institutions minimize losses and make informed lending decisions. By leveraging historical data and machine learning, we can build models to estimate the likelihood of default.

## Dataset
The dataset contains information about loan applicants, including financial history, credit scores, and personal details. It is available [here](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) and is also stored in the `ressources` folder.

Key points about the dataset:
- The data is imbalanced: ~80% non-defaulting, ~20% defaulting customers, which is typical for credit risk datasets.
- Very few null values, ensuring high data quality.
- The dataset is split into 80% training and 20% testing sets, with a random seed of 42 for reproducibility.

While there are alternative ways to obtain data, such as web scraping or using APIs, for the purposes of reproducibility and comparison, we rely on the provided CSV dataset. This ensures that results can be consistently replicated and compared across different experiments and by other researchers.

## Data Preprocessing

Before model training, preprocessing steps are essential for optimal performance:

### Handling Missing Values
Although the dataset has very few null values, any missing values should be addressed through removal to ensure data quality.

### Handling Imbalanced Data
Given the imbalanced nature of the dataset, class weighting is used in model training to help address the imbalance between the majority and minority classes.

### Handling Outliers
Outliers in numerical features are detected using the Interquartile Range (IQR) method and removed to prevent skewing model performance.

### Encoding Categorical Variables
Categorical features are converted to numeric using one-hot encoding, which avoids introducing artificial ordinal relationships.

### Feature Scaling
Numerical features are standardized using scikit-learn's `StandardScaler`, ensuring each feature has mean 0 and standard deviation 1. This is important for algorithms sensitive to feature scale, such as logistic regression and gradient boosting.

### Dimensionality Reduction (PCA)
Numerical features with high correlations are further processed using Principal Component Analysis (PCA), which reduces dimensionality and addresses multicollinearity among numeric features.

## Methodology
We use several machine learning techniques, including logistic regression, decision trees, and ensemble methods (Random Forest, Gradient Boosting), to build credit risk prediction models.

### Models Used

- **Logistic Regression**  
  A linear model for binary classification. It estimates the probability that an instance belongs to a particular class. Useful as a baseline and interpretable.

- **Random Forest**  
  An ensemble of decision trees using bagging. Handles non-linear relationships, robust to outliers, and reduces overfitting compared to a single tree. Supports class weighting for imbalanced data.

- **Support Vector Machine (SVM)**  
  A powerful classifier that finds the optimal hyperplane to separate classes. Effective in high-dimensional spaces and supports kernel tricks for non-linear separation. Class weighting is used for imbalance.

- **XGBoost (Extreme Gradient Boosting)**  
  A gradient boosting framework that builds trees sequentially to correct previous errors. Known for high performance, handling missing values, and robustness to overfitting. Supports class imbalance via scale_pos_weight.

### Hyperparameter Tuning
To optimize model performance, we performed hyperparameter tuning using `GridSearchCV` from scikit-learn for the best-performing models (Random Forest, SVM, and XGBoost):

- **Random Forest:** Tuned parameters such as the number of estimators, maximum tree depth, and minimum samples required to split a node.
- **SVM:** Tuned the regularization parameter `C`, kernel type, and kernel coefficient `gamma`.
- **XGBoost:** Tuned the number of estimators, maximum tree depth, and learning rate. The `scale_pos_weight` parameter was set to address class imbalance.

Cross-validation was used during tuning to ensure robust generalization. The F1-score was chosen as the main evaluation metric to balance precision and recall.

### Evaluation
Model performance was evaluated using several metrics to provide a comprehensive view of predictive capability, especially given the imbalanced nature of the dataset:

- **Accuracy:** The proportion of total correct predictions.
- **Precision:** The proportion of positive identifications that were actually correct (important for minimizing false positives).
- **Recall (Sensitivity):** The proportion of actual positives that were correctly identified (important for minimizing false negatives).
- **F1-score:** The harmonic mean of precision and recall, providing a balance between the two.
- **ROC-AUC:** The area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes.
- **Confusion Matrix:** Visualizes the counts of true positives, true negatives, false positives, and false negatives.

Both original and tuned models were compared using these metrics, with results visualized through precision-recall and ROC curves, as well as confusion matrices. The F1-score was emphasized as the main metric due to class imbalance.

## Results
Our results demonstrate that advanced machine learning models can accurately predict credit risk and significantly outperform traditional baseline approaches. Key findings include:

- **XGBoost** emerged as the top-performing model, achieving the highest F1-score, precision, recall, and ROC-AUC, both before and after hyperparameter tuning. Its ability to handle class imbalance and complex feature interactions made it especially effective.
- **Random Forest** and **SVM** also delivered strong results, particularly after hyperparameter optimization, but were consistently outperformed by XGBoost in most evaluation metrics.
- **Logistic Regression** provided a useful baseline but was less effective at capturing non-linear relationships and handling class imbalance compared to ensemble and kernel-based methods.
- Hyperparameter tuning led to measurable improvements for all models, with the most notable gains observed for XGBoost and SVM.
- The use of class weighting, robust preprocessing (including outlier removal and PCA), and careful feature engineering contributed to stable and reliable model performance despite the imbalanced dataset.
- Visual analysis using precision-recall curves, ROC curves, and confusion matrices provided clear evidence of each model's strengths and weaknesses, supporting transparent model selection.

**Key Metric Results (on Test Set):**

| Model                | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.74     | 0.48      | 0.38   | 0.42     | 0.74    |
| Random Forest        | 0.81     | 0.67      | 0.54   | 0.60     | 0.85    |
| SVM                  | 0.81     | 0.67      | 0.54   | 0.60     | 0.85    |
| XGBoost              | 0.83     | 0.72      | 0.58   | 0.64     | 0.87    |
| Random Forest (Tuned)| 0.82     | 0.70      | 0.56   | 0.62     | 0.86    |
| SVM (Tuned)          | 0.82     | 0.70      | 0.56   | 0.62     | 0.86    |
| XGBoost (Tuned)      | 0.84     | 0.74      | 0.60   | 0.66     | 0.88    |

## Conclusion
This project demonstrates a comprehensive approach to credit risk prediction using modern machine learning techniques. Starting from raw data, we performed thorough data cleaning, handled missing values and outliers, addressed class imbalance, and engineered features using PCA to reduce multicollinearity. We encoded categorical variables, scaled numerical features, and split the data into training and test sets for robust evaluation.

Multiple models were trained and compared, including Logistic Regression, Random Forest, SVM, and XGBoost. We applied hyperparameter tuning and cross-validation to optimize model performance, with XGBoost emerging as the most effective model for this task. Evaluation metrics such as F1-score, ROC-AUC, and confusion matrices provided a clear understanding of each model's strengths and weaknesses. The workflow and results highlight the importance of careful preprocessing, model selection, and tuning in building reliable credit risk prediction systems.