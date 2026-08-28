# ============================================================
# CREDIT CARD FRAUD DETECTION
# ============================================================

import os
import random

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Input,
    BatchNormalization,
    Dropout,
)

from tensorflow.keras.regularizers import l1_l2

from tensorflow.keras.optimizers import (
    Adam,
    RMSprop,
    SGD,
)

from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
OPTUNA_SEED = 21
N_TRIALS = 5

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "creditcard.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fraud_detection_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "fraud_scaler.joblib"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "fraud_features.joblib"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"""
Dataset not found:

{DATA_PATH}

Please place creditcard.csv in the
same folder as this Python file.
"""
    )


df = pd.read_csv(DATA_PATH)


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset description:")
print(df.describe())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass distribution percentage:")
print(
    df["Class"].value_counts(
        normalize=True
    ) * 100
)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# VALIDATION
# ============================================================

if "Class" not in df.columns:

    raise ValueError(
        "Target column 'Class' was not found."
    )


if df.isnull().sum().sum() > 0:

    raise ValueError(
        "Dataset contains missing values."
    )


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


print("\nFeature shape:", X.shape)
print("Target shape :", y.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=SEED,

    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(

    X_train,
    y_train,

    test_size=0.20,

    random_state=SEED,

    stratify=y_train
)


print("\nAfter validation split:")

print(
    "Training   :",
    X_train.shape
)

print(
    "Validation :",
    X_val.shape
)

print(
    "Testing    :",
    X_test.shape
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# SMOTE
# ============================================================

smote = SMOTE(
    random_state=SEED
)


X_train_res, y_train_res = (
    smote.fit_resample(
        X_train_scaled,
        y_train
    )
)


print("\n" + "=" * 60)
print("SMOTE RESULT")
print("=" * 60)

print("\nBefore SMOTE:")
print(
    y_train.value_counts()
)

print("\nAfter SMOTE:")
print(
    pd.Series(
        y_train_res
    ).value_counts()
)

print(
    "\nResampled training shape:",
    X_train_res.shape
)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    learning_rate = trial.suggest_float(

        "learning_rate",

        1e-4,
        1e-2,

        log=True
    )


    n_layers = trial.suggest_int(

        "n_layers",

        1,
        4
    )


    optimizer_name = trial.suggest_categorical(

        "optimizer",

        [
            "Adam",
            "RMSprop",
            "SGD"
        ]
    )


    activation = trial.suggest_categorical(

        "activation",

        [
            "tanh",
            "relu"
        ]
    )


    batch_size = trial.suggest_categorical(

        "batch_size",

        [
            32,
            64,
            128,
            256,
            512
        ]
    )


    # --------------------------------------------------------
    # BUILD TRIAL MODEL
    # --------------------------------------------------------

    trial_model = Sequential()


    trial_model.add(
        Input(
            shape=(
                X_train_res.shape[1],
            )
        )
    )


    for i in range(n_layers):

        units = trial.suggest_int(

            f"units{i}",

            8,
            96
        )


        dropout = trial.suggest_float(

            f"dropout{i}",

            0.0,
            0.5
        )


        reg = trial.suggest_float(

            f"reg{i}",

            1e-5,
            1e-2,

            log=True
        )


        trial_model.add(

            Dense(

                units,

                activation=activation,

                kernel_regularizer=l1_l2(

                    l1=reg,

                    l2=reg
                )
            )
        )


        trial_model.add(
            BatchNormalization()
        )


        trial_model.add(
            Dropout(dropout)
        )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    trial_model.add(

        Dense(

            1,

            activation="sigmoid"
        )
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    if optimizer_name == "Adam":

        optimizer = Adam(
            learning_rate=learning_rate
        )

    elif optimizer_name == "RMSprop":

        optimizer = RMSprop(
            learning_rate=learning_rate
        )

    else:

        optimizer = SGD(
            learning_rate=learning_rate
        )


    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    trial_model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping = EarlyStopping(

        monitor="val_recall",

        mode="max",

        patience=5,

        restore_best_weights=True
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = trial_model.fit(

        X_train_res,

        y_train_res,

        epochs=30,

        batch_size=batch_size,

        validation_data=(

            X_val_scaled,

            y_val
        ),

        callbacks=[
            early_stopping
        ],

        verbose=0
    )


    return max(
        history.history["val_recall"]
    )


# ============================================================
# OPTUNA
# ============================================================

print("\n" + "=" * 60)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 60)


study = optuna.create_study(

    direction="maximize",

    sampler=optuna.samplers.TPESampler(

        seed=OPTUNA_SEED
    )
)


study.optimize(

    objective,

    n_trials=N_TRIALS,

    show_progress_bar=True
)


print("\nBest parameters:")

print(
    study.best_params
)


print(
    "\nBest validation recall:",
    study.best_value
)


# ============================================================
# FINAL MODEL
# ============================================================

model = Sequential()


model.add(

    Input(

        shape=(
            X_train_res.shape[1],
        )
    )
)


# ============================================================
# FIRST HIDDEN LAYER
# ============================================================

model.add(

    Dense(

        37,

        activation="relu",

        kernel_initializer="he_normal",

        kernel_regularizer=l1_l2(

            l1=0.0004296403664910425,

            l2=0.0004296403664910425
        )
    )
)


model.add(
    BatchNormalization()
)


model.add(
    Dropout(
        0.23007016689648802
    )
)


# ============================================================
# SECOND HIDDEN LAYER
# ============================================================

model.add(

    Dense(

        27,

        activation="relu",

        kernel_initializer="he_normal",

        kernel_regularizer=l1_l2(

            l1=0.001513747182776292,

            l2=0.001513747182776292
        )
    )
)


model.add(
    BatchNormalization()
)


model.add(
    Dropout(
        0.39993416346948824
    )
)


# ============================================================
# OUTPUT LAYER
# ============================================================

model.add(

    Dense(

        1,

        activation="sigmoid",

        kernel_initializer="glorot_uniform"
    )
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=RMSprop(

        learning_rate=0.00330069279804087
    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)


model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_recall",

    mode="max",

    patience=5,

    restore_best_weights=True
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining final model...")


history = model.fit(

    X_train_res,

    y_train_res,

    epochs=30,

    batch_size=32,

    validation_data=(

        X_val_scaled,

        y_val
    ),

    callbacks=[
        early_stopping
    ],

    verbose=2
)


# ============================================================
# TEST PREDICTION
# ============================================================

y_prob = model.predict(

    X_test_scaled,

    verbose=0

).ravel()


y_pred = (

    y_prob >= 0.5

).astype(int)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred
)


precision = precision_score(

    y_test,

    y_pred,

    zero_division=0
)


recall = recall_score(

    y_test,

    y_pred,

    zero_division=0
)


f1 = f1_score(

    y_test,

    y_pred,

    zero_division=0
)


roc_auc = roc_auc_score(

    y_test,

    y_prob
)


print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")


print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# TRAINING LOSS
# ============================================================

plt.figure(

    figsize=(8, 5)
)


plt.plot(

    history.history["loss"],

    label="Training Loss"
)


plt.plot(

    history.history["val_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(

    y_test,

    y_prob
)


plt.figure(

    figsize=(8, 5)
)


plt.plot(

    fpr,

    tpr,

    label=f"ROC-AUC = {roc_auc:.4f}"
)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

print("\nSaving model files...")


model.save(
    MODEL_PATH
)


joblib.dump(

    scaler,

    SCALER_PATH
)


joblib.dump(

    list(X.columns),

    FEATURES_PATH
)


# ============================================================
# CONFIRM FILES
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED SUCCESSFULLY")
print("=" * 60)


print(
    f"Model    : {MODEL_PATH}"
)

print(
    f"Scaler   : {SCALER_PATH}"
)

print(
    f"Features : {FEATURES_PATH}"
)


print(
    "\nTraining completed successfully!"
)