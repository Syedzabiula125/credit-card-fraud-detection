# 💳 Credit Card Fraud Detection

An end-to-end **Machine Learning and Deep Learning project** that detects potentially fraudulent credit card transactions using a **Neural Network** and provides an interactive prediction interface using **Streamlit**.

---

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a small portion of total transactions.

This project builds a fraud detection pipeline that:

* Performs data preprocessing
* Splits the dataset into training, validation, and testing sets
* Applies feature scaling using `StandardScaler`
* Handles class imbalance using `SMOTE`
* Uses `Optuna` for hyperparameter optimization
* Builds a Neural Network using TensorFlow/Keras
* Evaluates the model using multiple classification metrics
* Saves the trained model and preprocessing objects
* Deploys the model through an interactive Streamlit application

---

## 🎯 Problem Statement

The goal of this project is to develop a machine learning system capable of identifying whether a credit card transaction is:

* ✅ **Legitimate**
* 🚨 **Fraudulent**

The system produces a fraud probability and classifies the transaction based on a predefined probability threshold.

---

## 🧠 Machine Learning Approach

### 1. Data Preprocessing

The dataset is separated into:

```text
Features (X)
        ↓
Target (y)
```

The target variable is:

```text
Class
```

where:

```text
0 → Legitimate Transaction
1 → Fraudulent Transaction
```

---

### 2. Train / Validation / Test Split

The dataset is divided into:

```text
Dataset
   │
   ├── Training Data
   ├── Validation Data
   └── Testing Data
```

Stratified splitting is used to maintain the fraud/legitimate class distribution.

---

### 3. Feature Scaling

`StandardScaler` is applied to normalize the numerical features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)
```

The scaler fitted on the training data is reused during prediction.

---

### 4. Handling Class Imbalance

Fraudulent transactions are significantly fewer than legitimate transactions.

To address this imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied only to the training data.

```text
Original Training Data
        ↓
      SMOTE
        ↓
Balanced Training Data
```

This helps the neural network learn patterns associated with fraudulent transactions.

---

## 🤖 Neural Network

The project uses a **TensorFlow/Keras Neural Network** for binary classification.

### Architecture

```text
Input Features
      ↓
Dense Layer
37 Neurons
ReLU
      ↓
Batch Normalization
      ↓
Dropout
      ↓
Dense Layer
27 Neurons
ReLU
      ↓
Batch Normalization
      ↓
Dropout
      ↓
Output Layer
1 Neuron
Sigmoid
```

### Configuration

**Optimizer:**

```text
RMSprop
```

**Loss Function:**

```text
Binary Crossentropy
```

**Output Activation:**

```text
Sigmoid
```

The sigmoid output represents the probability of a transaction being fraudulent.

---

## 🔍 Fraud Probability

The model produces a value between `0` and `1`.

For example:

```text
0.15 → 15% fraud probability
0.62 → 62% fraud probability
0.91 → 91% fraud probability
```

The application uses a threshold of `0.5`.

```text
Probability >= 0.5
        ↓
   Fraud / High Risk

Probability < 0.5
        ↓
   Legitimate / Low Risk
```

---

## ⚙️ Hyperparameter Optimization

**Optuna** is used to search for suitable neural network hyperparameters.

The optimization explores parameters such as:

* Learning rate
* Number of hidden layers
* Number of neurons
* Dropout rate
* Regularization
* Activation function
* Optimizer
* Batch size

The optimization objective focuses on **validation recall**, which is particularly important for fraud detection because missing fraudulent transactions can be costly.

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

### Why Recall?

In fraud detection, correctly identifying fraudulent transactions is especially important.

A model with high accuracy but poor recall may still miss many fraudulent transactions.

Therefore, this project considers **precision, recall, F1-score and ROC-AUC** along with accuracy.

---

## 🖥️ Streamlit Application

An interactive **Streamlit web application** is included for testing the trained model.

### Application Features

The dashboard contains:

### 🏠 Dashboard

Displays:

* Total transactions
* Fraud transactions
* Legitimate transactions
* Fraud percentage
* Transaction distribution
* Fraud percentage visualization

### 🔍 Fraud Prediction

Users can enter transaction feature values and receive:

```text
Fraud Probability
        ↓
Risk Classification
```

Example:

```text
Fraud Probability: 87.35%

🚨 HIGH FRAUD RISK
```

or:

```text
Fraud Probability: 12.42%

✅ LOW FRAUD RISK
```

### 📊 Model Performance

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

### ⚙️ Model Information

Displays:

* Neural network architecture
* Machine learning pipeline
* Dataset statistics
* Project objective

---

## 🛠️ Technologies Used

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Programming language         |
| Pandas       | Data manipulation            |
| NumPy        | Numerical computation        |
| Scikit-learn | Preprocessing and evaluation |
| SMOTE        | Handling class imbalance     |
| TensorFlow   | Deep learning                |
| Keras        | Neural network development   |
| Optuna       | Hyperparameter optimization  |
| Matplotlib   | Visualization                |
| Plotly       | Interactive visualization    |
| Streamlit    | Web application              |
| Joblib       | Saving preprocessing objects |

---

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    └── fraudshield.png
```

### Important Generated Files

The training script generates:

```text
fraud_detection_model.keras
fraud_scaler.joblib
fraud_features.joblib
```

These files are used by the Streamlit application for prediction.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Syedzabiula125/credit-card-fraud-detection.git
```

### 2. Navigate into the project

```bash
cd credit-card-fraud-detection
```

### 3. Create a virtual environment

```bash
python -m venv project
```

### 4. Activate the environment

### Windows

```bash
project\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run:

```bash
python model.py
```

The training pipeline will:

```text
Load Dataset
     ↓
Preprocessing
     ↓
Train / Validation / Test Split
     ↓
StandardScaler
     ↓
SMOTE
     ↓
Optuna Optimization
     ↓
Neural Network Training
     ↓
Model Evaluation
     ↓
Save Model
```

The following files will be generated:

```text
fraud_detection_model.keras
fraud_scaler.joblib
fraud_features.joblib
```

---

## 🌐 Run the Streamlit Application

After training the model:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually:

```text
http://localhost:8501
```

---

## 🔄 Complete Project Workflow

```text
                 Credit Card Dataset
                         │
                         ↓
                 Data Preprocessing
                         │
                         ↓
              Train / Validation / Test
                         │
                         ↓
                  StandardScaler
                         │
                         ↓
                       SMOTE
                         │
                         ↓
                  Optuna Tuning
                         │
                         ↓
                 Neural Network
                         │
                         ↓
                    Evaluation
                         │
                         ↓
                 Save Model Files
                         │
                         ↓
                 Streamlit Application
                         │
                         ↓
                 User Transaction
                         │
                         ↓
                    Preprocessing
                         │
                         ↓
                    Prediction
                         │
                         ↓
                 Fraud Probability
                         │
                         ↓
              Fraud Risk Classification
```

---

## 🔐 Important Note

The `V1`–`V28` features in the dataset are anonymized/PCA-transformed features. Their original real-world meanings are not directly available in the dataset.

`Amount` represents the transaction amount, while `Class` is the target variable.

```text
V1 – V28  → Anonymized features
Amount    → Transaction amount
Class     → Target
```

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Handling highly imbalanced datasets
* Data preprocessing
* Feature scaling
* SMOTE
* Neural Networks
* Hyperparameter optimization
* Model evaluation
* Fraud detection
* Model serialization
* Streamlit deployment
* Building an end-to-end ML application

---

## 👨‍💻 Author

**Syed Zabiulla**

GitHub:
https://github.com/Syedzabiula125

---

## ⭐ Future Improvements

Possible improvements include:

* Threshold optimization based on precision-recall tradeoff
* Real-time transaction monitoring
* Additional model comparison
* Explainable AI using SHAP
* Cloud deployment
* API-based prediction service
* Automated model retraining

---

⭐ If you find this project useful, consider giving the repository a star!
