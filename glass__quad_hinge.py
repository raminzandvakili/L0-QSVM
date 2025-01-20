from datetime import datetime
import logging
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lib.data import Data, get_data
from lib.helpers import setup_logging, log_eval_stats
from lib.models import Model, MinAlg, LossType, QuadraticSvmClassifier, SparseQuadraticSvmClassifier


################ CONFIGURATION ################
DATASET = Data.GLASS
MODEL = Model.QUADRATIC_SVM
LOSS_TYPE = LossType.HINGE
MIN_ALG = MinAlg.CVXPY
N_TESTS = 100
###############################################


RESULTS_DIR = f"{DATASET.value}/{MODEL.value}/{LOSS_TYPE.value}/{MIN_ALG}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

os.makedirs(f"{RESULTS_DIR}/images", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)

setup_logging(log_file=f"{RESULTS_DIR}/logs.log")

# HEART DISEASE DATASET (UCI)
logging.info(RESULTS_DIR)
logging.info(MODEL.value)
logging.info("=" * 40)
X, y = get_data(DATASET)

# Split data into training and testing sets for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


if MODEL == Model.LINEAR_KERNEL_SVM:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='linear'))
    ])

    param_grid = {'svm__C': loguniform(1e-6, 1e3)}


elif MODEL == Model.QUADRATIC_KERNEL_SVM:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='poly', degree=2))
    ])

    param_grid = {
                'svm__C': loguniform(1e-6, 1e3),
                'svm__gamma': loguniform(1e-6, 1e1),
                'svm__coef0': loguniform(1e-6, 1e1)
                }
    
elif MODEL == Model.POLYNOMIAL_KERNEL_SVM:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='poly'))
    ])

    param_grid = {
                'svm__C': loguniform(1e-6, 1e3),
                'svm__gamma': loguniform(1e-6, 1e1),
                'svm__coef0': loguniform(1e-6, 1e1),
                'svm__degree': [2, 3, 4, 5]
                }

elif MODEL == Model.SPARSE_QUADRATIC_SVM:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('sparse_quad_svm', SparseQuadraticSvmClassifier(
            rho=0.1,
            epsilon_inner=1e-2, 
            epsilon_outer=1e-1,
            loss_type=LOSS_TYPE.value,
            min_alg=MIN_ALG.value,
            gd_n_epochs=1000, 
            verbose=True))
    ])
    
    param_grid = {
            'sparse_quad_svm__mu': loguniform(1e-2, 1e2),
            'sparse_quad_svm__k': range(1, 2 * X_train.shape[1]),
            }
    
    
elif MODEL == Model.QUADRATIC_SVM:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('quad_svm', QuadraticSvmClassifier(max_iters=1000, verbose=False, batch_size=128, alpha=0))
    ])

    param_grid = {
                'quad_svm__mu': loguniform(1e-3, 1e3),
                'quad_svm__lambda_': loguniform(1e-3, 1e3),
                # 'quad_svm__alpha': loguniform(1e-3, 1e3),
                }

# Set up StratifiedKFold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search with cross-validation
random_search = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, n_iter=N_TESTS, verbose=0)
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# store the best model
joblib.dump(best_model, f"{RESULTS_DIR}/models/sparse_quadratic_svm_best_model.pkl")

# Make predictions on the test set and get probabilities
y_pred = best_model.predict(X_test)

logging.info("\nClassification Report with Best Threshold:")
logging.info(classification_report(y_test, y_pred))

# Summary of the best model's parameters
logging.info(f"Best Model Parameters: {random_search.best_params_}")

log_eval_stats(X, y, pipeline, param_grid, N_TESTS)
