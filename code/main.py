# main.py
import warnings

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_management import DataProcessor, remove_outliers
from visualization import DataVisualizer
from modeling import (RandomForestBaseline, RandomForestWithSMOTE, 
                      RandomForestWithRandomUnderSampling, RandomForestHyperTune,
                      AdaBoostBaseline, AdaBoostHyperTune, XGBoostBaseline, XGBoostHyperTune, KNNBaseline,KNNHyperTune)
from eda import LoanDataEDA
from sklearn.model_selection import train_test_split
import os

# Step 1: Data Management
# Dynamically find the CSV file in the main repository
project_folder = os.path.dirname(os.path.abspath(__file__))
csv_file = next(file for file in os.listdir(project_folder) if file.endswith(".csv"))

data_processor = DataProcessor(os.path.join(project_folder, csv_file))
data_processor.read_data()
data_processor.preprocess_data()

# Set specific thresholds for each column
thresholds = {
    'annual_inc': {'lower': data_processor.data["annual_inc"].quantile(0.005), 'upper': data_processor.data["annual_inc"].quantile(0.995)},
    'dti': {'upper': 45},
    'acc_now_delinq': {'upper': 6},
    'delinq_2yrs': {'upper': 35},
    'open_acc': {'upper': 80},
    'pub_rec': {'upper': 20},
    'revol_util': {'upper': 150},
    'mort_acc': {'upper': 30},
    'pct_tl_nvr_dlq': {'lower': 10},
    'tax_liens': {'upper': 60}
}

# Visualize outliers before and after removal
for column, column_thresholds in thresholds.items():
    # Apply outlier removal function for each column
    data_processor.data, outlier_count = remove_outliers(data_processor.data, column, column_thresholds.get('lower'), column_thresholds.get('upper'), drop_outliers=True)

    print(f"{column}: {outlier_count} outliers dropped.")

 # Pass the transformed data directly
data_processor.apply_onehot_encoding()
data_processor.split_data()

print()

# Now, you can access the training and testing sets
X_train, y_train = data_processor.X_train, data_processor.y_train
X_val, y_val = data_processor.X_val, data_processor.y_val
X_test, y_test = data_processor.X_test, data_processor.y_test


# For example, you can print the shape of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
print("Testing set shape:", X_val.shape, y_val.shape)


rf_baseline_model = RandomForestBaseline(X_train, y_train, X_test, y_test)
rf_baseline_model.train_model()
rf_baseline_model.evaluate_model("Random Forest Baseline")

input("Press Enter to continue...")

# Initialize and train the Random Forest model with SMOTE
rf_smote_model = RandomForestWithSMOTE(X_train, y_train, X_test, y_test)
rf_smote_model.apply_smote()
rf_smote_model.train_model()
rf_smote_model.evaluate_model("Random Forest with SMOTE")

input("Press Enter to continue...")

# Initialize and train the Random Forest model with Random Under Sampling
rf_undersample_model = RandomForestWithRandomUnderSampling(X_train, y_train, X_test, y_test)
rf_undersample_model.apply_random_under_sampling()
rf_undersample_model.train_model()
rf_undersample_model.evaluate_model("Random Forest with Random Under Sampling")

input("Press Enter to continue...")

# Define the parameter grid for hyperparameter tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [2, 4]
}

# Perform hyperparameter tuning for RandomForest model
rf_hyper_tuner = RandomForestHyperTune(X_train, y_train, X_val, y_val, X_test, y_test)
rf_hyper_tuner.hyperparameter_tuning(param_grid_rf)
rf_hyper_tuner.train_model()
rf_hyper_tuner.evaluate_model("RandomForestHyperTune", X_test, y_test)


input("Press Enter to continue...")

ada_baseline_model = AdaBoostBaseline(X_train, y_train, X_test, y_test)
ada_baseline_model.train_model()
ada_baseline_model.evaluate_model("AdaBoost Baseline")

input("Press Enter to continue...")

# Define the parameter grid for hyperparameter tuning
param_grid_ada = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    # You can include other base_estimator parameters here
}

ada_hyper_tuner = AdaBoostHyperTune(X_train, y_train, X_val, y_val, X_test, y_test)
ada_hyper_tuner.hyperparameter_tuning(param_grid_ada)
ada_hyper_tuner.train_model()
ada_hyper_tuner.evaluate_model("AdaBoostHyperTune")

input("Press Enter to continue...")

# Create an instance of XGBoostBaseline
xgb_baseline_model = XGBoostBaseline(X_train, y_train, X_test, y_test)
xgb_baseline_model.train_model()
xgb_baseline_model.evaluate_model("XGBoost Baseline")

input("Press Enter to continue...")

# Define the parameter grid for hyperparameter tuning
param_grid_xgb = {
    'max_depth': [3, 10],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

# Perform hyperparameter tuning for XGBoost model
xgb_hyper_tuner = XGBoostHyperTune(X_train, y_train, X_val, y_val, X_test, y_test)
xgb_hyper_tuner.hyperparameter_tuning(param_grid_xgb)
xgb_hyper_tuner.train_model()
xgb_hyper_tuner.evaluate_model("XGBoostHyperTune", X_test, y_test)

input("Press Enter to continue...")

# Initialize the KNNBaseline model
knn_baseline_model = KNNBaseline(X_train, y_train, X_test, y_test)
knn_baseline_model.train_model()
knn_baseline_model.evaluate_model("KNN Baseline")

input("Press Enter to continue...")

# # Initialize the KNNWithRandomUnderSampling model
# knn_under_sampling_model = KNNWithRandomUnderSampling(X_train, y_train, X_test, y_test)
# knn_under_sampling_model.apply_random_under_sampling()
# knn_under_sampling_model.train_model()
# knn_under_sampling_model.evaluate_model("KNN with Random Under Sampling")

# input("Press Enter to continue...")

# # Initialize the KNNWithSMOTE model
# knn_smote_model = KNNWithSMOTE(X_train, y_train, X_test, y_test)
# knn_smote_model.apply_smote()
# knn_smote_model.train_model()
# knn_smote_model.evaluate_model("KNN with SMOTE")


input("Press Enter to continue...")

# Define the parameter grid for hyperparameter tuning
param_grid_knn = {
    'n_neighbors': [ 5, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}

knn_hyper_tuner = KNNHyperTune(X_train, y_train, X_val, y_val, X_test, y_test)
knn_hyper_tuner.hyperparameter_tuning(param_grid_knn)
knn_hyper_tuner.train_model()
knn_hyper_tuner.evaluate_model("KNN HyperTune", X_test, y_test)

#####################################

#####################################

