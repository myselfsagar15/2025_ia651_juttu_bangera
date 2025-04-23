# Credit Card Fraud Detection Using Supervised Learning

##  Overview
This project tackles the challenge of detecting fraudulent credit card transactions using a suite of supervised machine learning algorithms. Given the extreme class imbalance in the dataset (~0.17% fraud), our focus is on optimizing recall while maintaining high precision.

---

##  Dataset

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows**: 284,807
- **Features**:
  - PCA-transformed: `V1` to `V28`
  - Others: `Amount`, `Time`
  - Target: `Class` (0 = normal, 1 = fraud)
- **Fraud Ratio**: 492 frauds (~0.17%)

---

##  Preprocessing

- Dropped `Time` feature
- Normalized `Amount` using `StandardScaler`
- Added `log_amount` for better variance control
- Stratified train-test split (70-30) to preserve fraud ratio

---

##  Exploratory Data Analysis (EDA)

- KDE plots showed distinguishable patterns for features like `V24`, `V10`, and `V14`
- Fraud transactions had a distinct range in `Amount`, often lower than normal
- Extreme skew in class distribution confirmed

---

##  Models Implemented

### 1. Logistic Regression
- **Precision (Fraud)**: 0.83
- **Recall (Fraud)**: 0.65
- **F1 Score**: 0.73
- **ROC-AUC**: 0.93

### 2. Decision Tree
- **Precision (Fraud)**: 0.78
- **Recall (Fraud)**: 0.72
- **F1 Score**: 0.75
- **ROC-AUC**: 0.86

### 3. Support Vector Machine (RBF)
- **Precision (Fraud)**: 0.97
- **Recall (Fraud)**: 0.62
- **F1 Score**: 0.76
- **ROC-AUC**: 0.94

### 4. Random Forest
- **Precision (Fraud)**: 0.94
- **Recall (Fraud)**: 0.83
- **F1 Score**: 0.88
- **ROC-AUC**: 0.95

### 5. XGBoost
- **Precision (Fraud)**: 0.94
- **Recall (Fraud)**: 0.76
- **F1 Score**: 0.84
- **ROC-AUC**: 0.93

---

##  Dimensionality Reduction & Clustering

- **t-SNE** visualization revealed clusters of fraud on the outskirts of the normal distribution
- **PCA** (2D) showed less separability between classes
- **KMeans** on PCA space failed to isolate fraud effectively

---

##  Summary of Results

| Model            | Precision | Recall | F1 Score | ROC-AUC |
|------------------|-----------|--------|----------|----------|
| Logistic Reg.    | 0.83      | 0.65   | 0.73     | 0.93     |
| Decision Tree    | 0.78      | 0.72   | 0.75     | 0.86     |
| SVM (RBF)        | 0.97      | 0.62   | 0.76     | 0.94     |
| **Random Forest**| **0.94**  | **0.83** | **0.88** | **0.95** |
| XGBoost          | 0.94      | 0.76   | 0.84     | 0.93     |

---

##  Conclusion

- **Random Forest** performed best overall with the highest balance between precision and recall
- **XGBoost** was close behind with great fraud capture and fewer false positives
- **SVM** had nearly perfect precision but lower recall — risky in fraud detection
- **Logistic Regression** gave strong results for a baseline model

---

##  Future Enhancements

- Apply **SMOTE** or **ADASYN** for class balancing
- Explore **ensemble methods** like stacking or blending

---

##  Authors

- **Ashish Varma** ,Sagar Bangera**, Clarkson University  
- **Course**: IA651 - Applied Machine Learning
