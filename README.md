# 📊 Credit Default Prediction using Random Forest

## 📌 Objective
The objective of this project is to build a machine learning model that predicts whether a customer will default on credit payments.  
This helps financial institutions assess credit risk and make informed lending decisions.

---

## 🧰 Tools & Technologies
- Python  
- Pandas  
- NumPy  
- scikit-learn  
- Matplotlib  
- Seaborn  
- Google Colab  

---

## 📊 Dataset
- **Source:** UCI Credit Card Default Dataset  
- **Target Variable:** `default.payment.next.month`  
  - `1` → Default  
  - `0` → No Default  

The dataset contains customer demographic information, credit history, and payment behavior.

---

## 🧠 Project Workflow

### 1️⃣ Setup
Prepared the development environment and structured the project for machine learning experimentation using Google Colab.

### 2️⃣ Installing Required Libraries
Installed essential Python libraries for data analysis, visualization, and machine learning.

### 3️⃣ Importing Required Libraries
Imported all required libraries to support data preprocessing, model training, and evaluation.

### 4️⃣ Import Data
Loaded the dataset and performed an initial inspection to understand data structure and feature types.

### 5️⃣ Analyze Missing Data
Checked for missing values to ensure data quality before training the model.

### 6️⃣ Downsample the Dataset
Handled class imbalance by downsampling the majority class to improve model fairness and performance.

### 7️⃣ One-Hot Encoding
Converted categorical variables into numerical format using one-hot encoding.

### 8️⃣ Split the Dataset
Split the dataset into training and testing sets to evaluate generalization performance.

### 9️⃣ Model Training & Evaluation
- Trained a **Random Forest Classifier**
- Evaluated performance using:
  - Accuracy score
  - Confusion matrix
  - Classification report

### 🔟 Hyperparameter Tuning
Optimized the model by tuning Random Forest hyperparameters to improve predictive performance.

---

## 📈 Results
- The Random Forest model achieved reliable accuracy.
- Feature importance analysis showed that payment history and bill amounts were strong predictors of credit default.
- The model demonstrated good classification capability on unseen data.

---

## 🧠 Key Learnings
- Handling imbalanced datasets  
- Feature engineering using one-hot encoding  
- Building and evaluating classification models  
- Hyperparameter tuning  
- Applying machine learning to real-world financial problems  

---

## ▶️ Run the Project
Open and run the notebook directly in Google Colab:

👉 **Open in Colab:**  
https://colab.research.google.com/github/Kartik8625/credit-default-prediction-random-forest/blob/main/credit_default_prediction.ipynb

---

## 📌 Conclusion
This project demonstrates an end-to-end machine learning pipeline for credit risk prediction, combining data preprocessing, model training, evaluation, and optimization.

---

## 👤 Author
**Kartik Inamdar**  
Aspiring Data Analyst | Machine Learning Enthusiast
