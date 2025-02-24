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
  

![Screenshot 2024-11-26 010056](https://github.com/user-attachments/assets/05056ef8-5d30-433d-a56f-20a9e935d96b)

- **Temporal Features:**
  
![Screenshot 2024-11-26 010109](https://github.com/user-attachments/assets/c2385102-08b4-4c56-8bf8-feb5ca1d61e3)

- **Spectral Features:**
  
![Screenshot 2024-11-26 010423](https://github.com/user-attachments/assets/a775df23-1b46-49e7-a2d8-c9eabd93127b)





### 📌 2. Dimensionality Reduction (PCA)

- **Cumulative Variance Explained:**

![Screenshot 2024-11-27 114328](https://github.com/user-attachments/assets/9ae5e077-11c3-43db-bb45-a196b0377690)

- **Data Projection After PCA:**

![Screenshot 2024-11-27 114241](https://github.com/user-attachments/assets/b4c0d8ac-6d75-4027-b4f9-0f9d69e36efa)

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

  ![Screenshot 2024-11-27 115522](https://github.com/user-attachments/assets/2adac253-72f3-44bc-844d-ebfec3d182fd)


- **Random Forest Classifier:**

  ![Screenshot 2024-11-27 114854](https://github.com/user-attachments/assets/4b1a1f8e-cca1-44de-bbab-b1465131d2f6)


- **K-Means Clustering:**

  ![Screenshot 2024-11-27 115937](https://github.com/user-attachments/assets/4ba547a3-6359-4deb-8fcc-8e443f7e30a2)


- **Birch Clustering:**

  ![Screenshot 2024-11-27 115949](https://github.com/user-attachments/assets/fa1de4db-ced1-45e9-a757-d1947b61b254)


---

## 📖 References

- **MIT-BIH Arrhythmia Database** for ECG signal data.
- **Random Forest & KNN** for supervised classification.
- **PCA & Clustering Methods** for dimensionality reduction and pattern detection.

---

## 📜 License

This project is licensed under the **MIT License**.
