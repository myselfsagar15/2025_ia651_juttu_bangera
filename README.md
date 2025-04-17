# Credit Card Fraud Detection Using Unsupervised Learning

## Abstract
Fraudulent transaction detection is a major challenge in real-time systems due to the scarcity and imbalance of labeled data. This project applies unsupervised learning techniques—Isolation Forest and Local Outlier Factor (LOF)—to detect anomalies in credit card transaction data. The methodology includes dimensionality reduction, exploratory data analysis, and performance benchmarking based on precision, recall, and F1-score.


## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total Records**: 284,807 transactions
- **Attributes**: 30 features including anonymized PCA components (V1 to V28), `Time`, `Amount`, and `Class` (0: Legitimate, 1: Fraudulent)

## Methodology

### Preprocessing
- Normalization using `StandardScaler` and `MinMaxScaler`
- Feature selection based on variance and correlation
- Dimensionality reduction using **UMAP** for better separability of anomalies in 2D space

### Exploratory Data Analysis
- Analyzed time-of-day patterns and distribution of principal components
- Key observations:
  - Fraudulent transactions occur more frequently during early hours (0–6)
  - Distinct distributions observed in features V1 to V9 between fraud and normal classes

## Data Visualization

Visual analytics were performed to understand feature distributions and identify patterns that differentiate fraud from legitimate transactions:

- **KDE (Kernel Density Estimate) Plots**:
  - Compared distributions for high-variance features such as `V1`, `V4`, and `V6`
  - Fraudulent transactions showed distinct non-Gaussian spread, especially in early morning hours (`Time`)
  ![image](https://github.com/user-attachments/assets/d6f23626-fe47-407a-801a-5e03c455b612)


- **Heatmap of Correlations**:
  - Highlighted strong negative/positive correlation between certain PCA components and class labels
  ![image](https://github.com/user-attachments/assets/3bd167d4-5d35-43da-9979-e3456bd967eb)


- **UMAP Projections**:
  - 2D visualization using UMAP demonstrated strong separability of outliers, validating its utility for anomaly detection
  ![image](https://github.com/user-attachments/assets/84c62416-0710-4637-b38a-5301f52815b3)


- **Histograms and Boxplots**:
  - Used to observe skewness and interquartile ranges of `Amount`, `Time`, and selected V-features
  ![image](https://github.com/user-attachments/assets/26d20d63-8702-4bd3-b95e-f920a6c507b7)


- **Class Distribution Plot**:
  - Displayed the extreme class imbalance (~0.17% fraud)
  ![image](https://github.com/user-attachments/assets/ba80132e-6b14-49ae-b9f6-a5ccb98c0143)


## Models Implemented

- **Isolation Forest** (`sklearn.ensemble.IsolationForest`)
  - Works by recursively partitioning data; fewer splits indicate anomalies

- **Local Outlier Factor (LOF)** (`sklearn.neighbors.LocalOutlierFactor`)
  - Detects local density deviations from neighbors

- **Evaluation Metrics**:
  - **Precision**: Proportion of correctly identified frauds out of total predicted frauds
  - **Recall**: Proportion of correctly identified frauds out of all true frauds
  - **F1-Score**: Harmonic mean of precision and recall

| Metric        | Isolation Forest | Local Outlier Factor |
|---------------|------------------|-----------------------|
| Precision     | ~0.09            | ~0.06                 |
| Recall        | ~0.30            | ~0.16                 |
| F1-Score      | ~0.14            | ~0.09                 |

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

- Time-based features and PCA components (e.g., V4, V6, V7) were effective in distinguishing fraud

- UMAP visualization revealed separable clusters for anomalies

- Isolation Forest yielded better recall, making it suitable for fraud detection in highly imbalanced datasets

- The methodology can be extended to real-time pipelines using streaming anomaly detection frameworks


## Acknowledgment

This project was developed as part of the academic course **IA651: Applied Machine Learning** at **Clarkson University** under the supervision of **Prof. Michael Gilbert**.

**Authors**  
Ashish Varma Juttu, Clarkson University  
Sagar Bangera, Clarkson University
