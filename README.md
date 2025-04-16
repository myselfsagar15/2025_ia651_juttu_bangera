# Credit Card Fraud Detection Using Unsupervised Learning

## Abstract
Fraudulent transaction detection is a major challenge in real-time systems due to the scarcity and imbalance of labeled data. This project applies unsupervised learning techniques—Isolation Forest and Local Outlier Factor (LOF)—to detect anomalies in credit card transaction data. The methodology includes dimensionality reduction, exploratory data analysis, and performance benchmarking based on precision, recall, and F1-score.


## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total Records**: 284,807 transactions
- **Attributes**: 30 features including anonymized PCA components (V1 to V28), `Time`, `Amount`, and `Class` (0: Legitimate, 1: Fraudulent)


## Methodology

### Preprocessing
- Normalization and scaling applied to numerical attributes
- Dimensionality reduction performed using Uniform Manifold Approximation and Projection (UMAP)

### Exploratory Data Analysis
- Analyzed time-of-day patterns and distribution of principal components
- Key observations:
  - Fraudulent transactions occur more frequently during early hours (0–6)
  - Distinct distributions observed in features V1 to V9 between fraud and normal classes

### Models Implemented
- **Isolation Forest**: An ensemble-based method that isolates anomalies
- **Local Outlier Factor**: A density-based method to detect local deviations


## Evaluation

| Metric        | Isolation Forest | Local Outlier Factor |
|---------------|------------------|-----------------------|
| Precision     | ~0.09            | ~0.06                |
| Recall        | ~0.30            | ~0.16                |
| F1-Score      | ~0.14            | ~0.09                |

- **Confusion Matrix** and **Classification Reports** provided for both models.
- Isolation Forest demonstrated better performance overall in terms of recall.


## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Clarkson-Applied-Data-Science/2025_ia651_juttu_bangera.git
   cd 2025_ia651_juttu_bangera
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook creditCard.ipynb
   ```


## Conclusion

Unsupervised learning is effective for fraud detection in the absence of labeled data. Time-based patterns and principal components such as V4 and V6 provide useful insights. Isolation Forest outperforms LOF in this context, offering a better balance between false positives and true fraud detection.


## Acknowledgment

This project was developed as part of the academic course **IA651: Applied Machine Learning** at **Clarkson University** under the supervision of **Prof. Michael Gilbert**.

**Authors**  
Ashish Varma Juttu, Clarkson University  
Sagar Bangera, Clarkson University
