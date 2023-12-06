# modeling.py

from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score,recall_score,f1_score,roc_curve, auc,roc_auc_score,confusion_matrix,classification_report)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestBaseline:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        self.rf_classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name):
        # Make predictions on the test set
        y_pred_proba = self.rf_classifier.predict_proba(self.X_test)
        y_pred = self.rf_classifier.predict(self.X_test)

        # Use the reusable metrics function
        evaluate_classification_metrics(
            self.y_test, y_pred, y_pred_proba, model_name
        )


class RandomForestWithSMOTE:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42
        )

    def apply_smote(self):
        # Apply SMOTE to the training set
        smote = SMOTE(random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
            self.X_train, self.y_train
        )

    def train_model(self):
        # Train the classifier on the resampled training set
        self.rf_classifier.fit(self.X_train_resampled, self.y_train_resampled)

    def evaluate_model(self, model_name):
        # Make predictions on the test set
        y_pred_proba = self.rf_classifier.predict_proba(self.X_test)
        y_pred = self.rf_classifier.predict(self.X_test)

        # Use the reusable metrics function
        evaluate_classification_metrics(
            self.y_test, y_pred, y_pred_proba, model_name
        )


class RandomForestWithRandomUnderSampling:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=4, min_samples_split=2, random_state=42
        )

    def apply_random_under_sampling(self):
        # Apply Random Under Sampling to the training set
        undersampler = RandomUnderSampler(random_state=42)
        self.X_train_undersampled, self.y_train_undersampled = undersampler.fit_resample(
            self.X_train, self.y_train
        )

    def train_model(self):
        # Train the classifier on the undersampled training set
        self.rf_classifier.fit(self.X_train_undersampled, self.y_train_undersampled)


    def evaluate_model(self, model_name):
        # Make predictions on the test set
        y_pred_proba = self.rf_classifier.predict_proba(self.X_test)
        y_pred = self.rf_classifier.predict(self.X_test)

        # Use the reusable metrics function
        evaluate_classification_metrics(
            self.y_test, y_pred, y_pred_proba, model_name
        )

class RandomForestHyperTune:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_rf_model = None

    def hyperparameter_tuning(self, param_grid):
        # Define a scorer for multiclass ROC AUC
        roc_auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)

        # Initialize the RandomForestClassifier
        rf_classifier = RandomForestClassifier(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf_classifier,
            param_grid=param_grid,
            scoring=roc_auc_scorer,
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        # Fit the grid search to the data
        grid_search.fit(self.X_train, self.y_train)

        # Print the best parameters and corresponding ROC AUC score
        print("Best Parameters:", grid_search.best_params_)
        print("Best ROC AUC Score:", grid_search.best_score_)

        # Set the best model
        self.best_rf_model = grid_search.best_estimator_

    def train_model(self):
        # Train the best model on the full training set
        self.best_rf_model.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name, X, y):
        # Make predictions on the specified set
        y_pred_proba = self.best_rf_model.predict_proba(X)
        y_pred = self.best_rf_model.predict(X)

        # Use the reusable metrics function
        evaluate_classification_metrics(y, y_pred, y_pred_proba, model_name)


class AdaBoostBaseline:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.ada_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        self.ada_classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name):
        # Make predictions on the test set
        y_pred_proba = self.ada_classifier.predict_proba(self.X_test)
        y_pred = self.ada_classifier.predict(self.X_test)

        # Use the reusable metrics function
        evaluate_classification_metrics(
            self.y_test, y_pred, y_pred_proba, model_name
        )

class AdaBoostHyperTune:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.best_ada_classifier = None

    def hyperparameter_tuning(self, param_grid):
        # Define the AdaBoostClassifier
        ada_classifier = AdaBoostClassifier(random_state=42)

        # Use accuracy as the scoring metric
        scorer = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=ada_classifier, param_grid=param_grid, scoring=scorer, cv=3, verbose=2, n_jobs=-1)

        # Fit the grid search to the data
        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters from the grid search
        self.best_params = grid_search.best_params_
        print("Best Parameters:", self.best_params)

    def train_model(self):
        # Initialize AdaBoostClassifier with the best parameters
        self.best_ada_classifier = AdaBoostClassifier(**self.best_params, random_state=42)

        # Train the classifier on the full training set
        self.best_ada_classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name):
        # Make predictions on the validation set
        y_val_pred = self.best_ada_classifier.predict(self.X_val)

        # Evaluate the performance on the validation set (you can use your desired metrics)
        accuracy = accuracy_score(self.y_val, y_val_pred)
        print(f"{model_name} Validation Accuracy:", accuracy)

        # Make predictions on the test set
        y_pred_proba = self.best_ada_classifier.predict_proba(self.X_test)
        y_pred = self.best_ada_classifier.predict(self.X_test)

        # Use the reusable metrics function
        evaluate_classification_metrics(
            self.y_test, y_pred, y_pred_proba, model_name
        )

class XGBoostBaseline:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.xgb_classifier = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        self.xgb_classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name):
        # Make predictions on the test set
        y_pred_proba = self.xgb_classifier.predict_proba(self.X_test)
        y_pred = self.xgb_classifier.predict(self.X_test)

        # Use the reusable metrics function
        evaluate_classification_metrics(
            self.y_test, y_pred, y_pred_proba, model_name
        )


class XGBoostHyperTune:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_xgb_model = None

    def hyperparameter_tuning(self, param_grid):
        # Define a scorer for multiclass F1 score
        f1_scorer = make_scorer(f1_score, average='macro')

        # Initialize the XGBClassifier
        xgb_classifier = xgb.XGBClassifier(objective='multi:softprob', random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_classifier,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        # Fit the grid search to the data
        grid_search.fit(self.X_train, self.y_train)

        # Print the best parameters and corresponding F1 score
        print("Best Parameters:", grid_search.best_params_)
        print("Best F1 Score:", grid_search.best_score_)

        # Set the best model
        self.best_xgb_model = grid_search.best_estimator_

    def train_model(self):
        # Train the best model on the full training set
        self.best_xgb_model.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name, X, y):
        # Make predictions on the specified set
        y_pred_proba = self.best_xgb_model.predict_proba(X)
        y_pred = self.best_xgb_model.predict(X)

        # Use the reusable metrics function
        evaluate_classification_metrics(y, y_pred, y_pred_proba, model_name)

class KNNBaseline:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        self.knn_classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name):
        # Make predictions on the test set
        y_pred_proba = self.knn_classifier.predict_proba(self.X_test)
        y_pred = self.knn_classifier.predict(self.X_test)

        # Use the reusable metrics function
        evaluate_classification_metrics(
            self.y_test, y_pred, y_pred_proba, model_name
        )

# class KNNWithRandomUnderSampling:
#     def __init__(self, X_train, y_train, X_test, y_test):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
#         self.knn_classifier = KNeighborsClassifier(n_neighbors=5)

#     def apply_random_under_sampling(self):
#         # Apply Random Under Sampling to the training set
#         undersampler = RandomUnderSampler(random_state=42)
#         self.X_train_undersampled, self.y_train_undersampled = undersampler.fit_resample(
#             self.X_train, self.y_train
#         )

#     def train_model(self):
#         # Train the k-Nearest Neighbors classifier on the undersampled training set
#         self.knn_classifier.fit(self.X_train_undersampled, self.y_train_undersampled)

#     def evaluate_model(self, model_name):
#         # Make predictions on the test set
#         y_pred_proba = self.knn_classifier.predict_proba(self.X_test)
#         y_pred = self.knn_classifier.predict(self.X_test)

#         # Use the reusable metrics function
#         evaluate_classification_metrics(
#             self.y_test, y_pred, y_pred_proba, model_name
#         )

# class KNNWithSMOTE:
#     def __init__(self, X_train, y_train, X_test, y_test):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
#         self.knn_classifier = KNeighborsClassifier(n_neighbors=5)

#     def apply_smote(self):
#         # Apply SMOTE to the training set
#         smote = SMOTE(random_state=42)
#         self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
#             self.X_train, self.y_train
#         )

#     def train_model(self):
#         # Train the k-Nearest Neighbors classifier on the resampled training set
#         self.knn_classifier.fit(self.X_train_resampled, self.y_train_resampled)

#     def evaluate_model(self, model_name):
#         # Make predictions on the test set
#         y_pred_proba = self.knn_classifier.predict_proba(self.X_test)
#         y_pred = self.knn_classifier.predict(self.X_test)

#         # Use the reusable metrics function
#         evaluate_classification_metrics(
#             self.y_test, y_pred, y_pred_proba, model_name
#         )


class KNNHyperTune:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_knn_model = None

    def hyperparameter_tuning(self, param_grid):
        # Define a scorer for multiclass ROC AUC
        roc_auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)

        # Initialize the KNeighborsClassifier
        knn_classifier = KNeighborsClassifier()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=knn_classifier,
            param_grid=param_grid,
            scoring=roc_auc_scorer,
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        # Fit the grid search to the data
        grid_search.fit(self.X_train, self.y_train)

        # Print the best parameters and corresponding ROC AUC score
        print("Best Parameters:", grid_search.best_params_)
        print("Best ROC AUC Score:", grid_search.best_score_)

        # Set the best model
        self.best_knn_model = grid_search.best_estimator_

    def train_model(self):
        # Train the best model on the full training set
        self.best_knn_model.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name, X, y):
        # Make predictions on the specified set
        y_pred_proba = self.best_knn_model.predict_proba(X)
        y_pred = self.best_knn_model.predict(X)

        # Use the reusable metrics function
        evaluate_classification_metrics(y, y_pred, y_pred_proba, model_name)


def evaluate_classification_metrics(y_true, y_pred, y_pred_proba, model_name):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(
        label_binarize(y_true, classes=np.unique(y_true)),
        y_pred_proba,
        average='weighted',
        multi_class='ovr'
    )

    # Print metrics with model name
    print(f"{model_name} Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # ROC-AUC Curve
    # Binarize the output for multiclass ROC curve
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute weighted-average ROC curve and ROC area
    fpr["weighted"], tpr["weighted"], _ = roc_curve(
        y_true_bin.ravel(), y_pred_proba.ravel()
    )
    roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})'
        )

    plt.plot(
        fpr["weighted"], tpr["weighted"], color='deeppink', linestyle=':', linewidth=4,
        label=f'Weighted-average ROC curve (area = {roc_auc["weighted"]:0.2f})'
    )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Additionally, print a comprehensive classification report with model name
    print(f"\nClassification Report for {model_name}:\n", classification_report(y_true, y_pred))