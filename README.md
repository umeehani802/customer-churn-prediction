# 🧠 Customer Churn Prediction using Machine Learning

This project predicts whether a customer will leave (churn) a bank using various machine learning models. By analyzing customer data, banks can proactively identify potential churners and take actions to retain them.

## 📖 Overview

Customer churn is a critical problem in the banking industry. The aim of this project is to:
- Explore and clean the customer dataset.
- Build and compare multiple classification models.
- Predict customer churn.
- Analyze important features driving churn.

## 📊 Dataset

The dataset contains 10,000+ records of bank customers with the following features:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (Target: 1 = Churn, 0 = No churn)

> ✅ No missing values  
> 🔍 Features encoded using Label Encoding and One-Hot Encoding where necessary.

## 🛠️ Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **Pandas, NumPy** – Data wrangling
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Model building & evaluation

  
## 🔁 Workflow

1. **Exploratory Data Analysis**
   - Checked distributions, relationships, and correlations.
2. **Data Cleaning**
   - Verified data types, filled missing values, handled duplicates.
3. **Feature Engineering**
   - Applied One-Hot and Label Encoding.
4. **Model Training**
   - Trained 5 different ML classifiers.
5. **Evaluation**
   - Compared accuracy scores and confusion matrices.
6. **Feature Importance**
   - Visualized top features contributing to churn.


## 🤖 Modeling

Models used:
- ✅ Decision Tree
- ✅ Random Forest
- ✅ Support Vector Machine (SVM)
- ✅ K-Nearest Neighbors (KNN)
- ✅ Logistic Regression (baseline)

All models were evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

## 📈 Results

| Model                 | Accuracy (%) |
|----------------------|--------------|
| Decision Tree        | 85.26        |
| Random Forest        | 86.45        |
| SVM                  | 86.13        |
| KNN                  | 82.74        |
| Logistic Regression  | 79.83        |

> 🎯 **Random Forest** performed the best overall.

## ✅ Conclusion

- **Geography**, **Age**, **Balance**, and **IsActiveMember** are key drivers of churn.
- **Random Forest Classifier** provided the highest accuracy and best generalization.
- With accurate churn predictions, banks can target high-risk customers with retention strategies.

> 🔮 *Future improvements could include class balancing, hyperparameter tuning, and deep learning models.*


