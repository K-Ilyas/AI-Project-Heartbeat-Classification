# Heartbeat Classification

## 📌 Project Overview

This project focuses on the classification of **heartbeat signals** using various **supervised** and **unsupervised** machine learning techniques. The dataset consists of ECG signals from the **MIT-BIH Arrhythmia** database, which have been preprocessed and segmented for classification.

### 🔹 Key Objectives:

- **Analyze correlations** between different features (statistical, temporal, and spectral).
- **Apply dimensionality reduction techniques** such as Principal Component Analysis (PCA).
- **Train supervised models** (Random Forest, K-Nearest Neighbors) for classification.
- **Compare unsupervised methods** (Birch, K-Means) for clustering.
- **Optimize hyperparameters** using Grid Search.
- **Evaluate model performance** with accuracy metrics and confusion matrices.

---

## 🛠 Installation Guide

### 1️⃣ Prerequisites

Ensure that you have the following installed:

- **Python 3.7+**
- Required Python libraries:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn
  ```

### 2️⃣ Cloning the Repository

```bash
git clone https://github.com/K-Ilyas/AI-Project-Heartbeat-Classification.git
cd heartbeat-classification
```

---

## 📜 Implementation Details

### 📌 1. Data Analysis and Feature Engineering

- **Correlation matrices** to analyze relationships between features.
- **Dimensionality reduction with PCA** to retain important components while reducing complexity.

**Correlation Matrices:**
- **Statistical Features:**
- **Temporal Features:**
- **Spectral Features:**

### 📌 2. Dimensionality Reduction (PCA)

- **Cumulative Variance Explained:**

- **Data Projection After PCA:**


### 📌 3. Supervised Learning Models

- **Random Forest** with optimized hyperparameters using Grid Search.
- **K-Nearest Neighbors (KNN)** for classification.

### 📌 4. Unsupervised Learning Methods

- **Birch clustering** for pattern detection.
- **K-Means clustering** for heartbeat segmentation.

### 📌 5. Model Evaluation

- **Accuracy scores** for supervised learning.
- **Silhouette scores** for clustering performance.
- **Confusion matrices** for classification analysis.

---

## 📊 Results and Visualizations

### 🔹 Supervised Learning Results

- **Random Forest Accuracy:** 70.26%
- **KNN Accuracy:** 73.2%

### 🔹 Unsupervised Learning Results

- **K-Means Accuracy:** 49.5%
- **Birch Accuracy:** 39.6%

### 🔹 Confusion Matrices

- **KNN Classifier:**

- **Random Forest Classifier:**

- **K-Means Clustering:**

- **Birch Clustering:**


---

## 📖 References

- **MIT-BIH Arrhythmia Database** for ECG signal data.
- **Random Forest & KNN** for supervised classification.
- **PCA & Clustering Methods** for dimensionality reduction and pattern detection.

---

## 📜 License

This project is licensed under the **MIT License**.
