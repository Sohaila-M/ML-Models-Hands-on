# 🧠 Machine Learning Models – Hands-On Labs

This repository contains my hands-on work from a machine learning course at National Telecommunication Institute (NTI) as a part of the big data engineering track , where I implemented a variety of ML models using real-world datasets. Each notebook covers the full pipeline: data preprocessing, model training, evaluation, and visualization where applicable.

> ⚙️ All projects are written in Python, primarily using **scikit-learn**, **matplotlib**, and **seaborn**. One notebook demonstrates usage with **PySpark**.

---

## 📁 Repository Structure

### 🔧 1. Data Preprocessing

* **`preprocessing_python.ipynb`**
  Basic data cleaning and preparation techniques (handling missing values, encoding, scaling, etc.) on a general dataset.

* **`Play_Store_Data.ipynb`**
  Full preprocessing pipeline on Google Play Store dataset followed by Random Forest implementation.

---

### 📈 2. Regression Models

* **`LinearRegression.ipynb`**
  Applied Linear Regression on `Salary_Data` to predict salaries based on experience.

* **`DecisionTreeRegression.ipynb`**
  Used Decision Tree Regression on `Position_Salaries` dataset with model visualization.

* **`Random_Forest_Regression.ipynb`**
  Implemented Random Forest Regression on the same `Position_Salaries` data.

---

### 🧪 3. Classification Models

* **`DecisionTreeClassifier.ipynb`**
  Decision Tree Classification on `Social_Network_Ads` dataset with visual analysis.

* **`KNN.ipynb`**
  K-Nearest Neighbors classification on `Social_Network_Ads`, including evaluation and comparison with Decision Tree.

* **`Random_Forest_Classification.ipynb`**
  Applied Random Forest Classification to `Social_Network_Ads` dataset and examined bootstrap samples used for trees.

* **`Heart_Disease_Classification_vs_Clustering.ipynb`**
  Preprocessed a heart disease dataset and compared classification (Random Forest) with clustering (KMeans) performance.

---

### 📊 4. Clustering Models

* **`Mall_Customers_Segmentation.ipynb`**
  Customer segmentation using K-Means on mall customer data, with visual cluster analysis.

---

### 🧬 5. Dimensionality Reduction

* **`PCA.ipynb`**
  Performed Principal Component Analysis on a heart dataset with 76 features, reducing it to 14 key components.

---

### ⚡ 6. Advanced & Deep Learning

* **`Spark-ML-LinearRegression&RandomForest.ipynb`**
  Linear Regression and Random Forest using **Apache Spark**'s `pyspark.ml` module for large-scale ML pipelines.

* **`ANN.ipynb`**
  Implemented a basic Artificial Neural Network (ANN) for binary classification on `Churn_Modelling` dataset.


## 🧰 Tools & Libraries Used

* Python, Jupyter Notebooks
* scikit-learn, pandas, matplotlib, seaborn
* PySpark (`pyspark.ml`)
* Keras / TensorFlow (for ANN)
