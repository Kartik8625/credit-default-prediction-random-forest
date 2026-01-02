📊 Credit Default Prediction using Random Forest
📌 Objective

The objective of this project is to build a machine learning model that predicts whether a customer will default on credit payments.
Such predictions help financial institutions assess credit risk and make informed lending decisions.

🧰 Tools & Technologies

Python

Pandas, NumPy

scikit-learn

Matplotlib, Seaborn

Google Colab

📊 Dataset

Source: UCI Credit Card Default Dataset

Target Variable: default.payment.next.month

1 → Defaulted

0 → Did Not Default

The dataset contains customer demographic information, credit limits, billing history, and repayment behavior.

🧠 Project Workflow

Setup & Environment Preparation
Configured the development environment using Google Colab.

Data Loading & Inspection
Loaded the dataset and analyzed its structure, feature types, and class distribution.

Missing Data Analysis
Checked for missing values to ensure data quality before model training.

Handling Class Imbalance
Addressed class imbalance using sampling strategies and class weighting.

Feature Engineering
Applied one-hot encoding to categorical variables to make them compatible with machine learning models.

Train-Test Split
Split the dataset into training and testing sets to evaluate generalization performance.

Model Training
Trained a Random Forest Classifier to capture non-linear patterns and feature interactions.

Model Evaluation
Evaluated the model using:

Accuracy

Precision, Recall

F1-Score

Confusion Matrix

Hyperparameter Tuning
Used RandomizedSearchCV with cross-validation to optimize key model parameters and improve robustness.

📈 Model Performance

Accuracy: 67.50%

F1-Score: 0.66

Class-wise Performance
Class	Precision	Recall	F1-Score
Did Not Default	0.68	0.70	0.69
Defaulted	0.67	0.65	0.66

The model demonstrates balanced performance across both classes, which is critical for credit risk assessment where misclassification costs are high.

🧠 Key Learnings

Handling imbalanced datasets in classification problems

Feature encoding for machine learning models

Evaluating models using F1-score instead of relying only on accuracy

Hyperparameter tuning with cross-validation

Applying machine learning to real-world financial risk problems

▶️ Run the Project

Open and run the notebook directly in Google Colab:

👉 Open in Colab
https://colab.research.google.com/github/Kartik8625/credit-default-prediction-random-forest/blob/main/credit_default_prediction.ipynb

📌 Conclusion

This project demonstrates an end-to-end machine learning pipeline for credit default prediction.
The Random Forest model achieved stable and balanced results, making it a reliable baseline for credit risk analysis and further feature engineering.

👤 Author

Kartik Inamdar
Aspiring Data Analyst | Machine Learning Enthusiast
