# visualization.py
import warnings

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_management import remove_outliers 
from eda import LoanDataEDA
from data_management import DataProcessor
import os

class DataVisualizer:
    @staticmethod
    def plot_correlation_heatmap(data):
        numerical_data = data.select_dtypes(include=['float64', 'int64'])
        # Calculate correlation matrix
        data_corr = numerical_data.corr()

        # Plot heatmap
        plt.figure(figsize=(12, 7))
        sns.heatmap(data_corr, annot=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()


    @staticmethod
    def visualize_outliers_before_after(data, columns=None, title_prefix="", thresholds=None):
        # Set the style of seaborn
        sns.set(style="whitegrid")

        # If specific columns are provided, create individual boxplots for before and after removal
        if columns and thresholds:
            for column in columns:
                # Create a subplot for before and after outlier removal
                plt.figure(figsize=(12, 7))

                # Before Outlier Removal
                plt.subplot(1, 2, 1)
                sns.boxplot(x=data[column], palette="Set2")
                plt.title(f"{title_prefix}: Before Outlier Removal - {column}", fontsize=16)

                # Create a copy of the data before outlier removal
                data_no_outliers = data.copy()

                # Remove outliers based on the specified thresholds
                data_no_outliers, _ = remove_outliers(data_no_outliers, column, thresholds.get('lower'), thresholds.get('upper'), drop_outliers=True)

                # After Outlier Removal
                plt.subplot(1, 2, 2)
                sns.boxplot(x=data_no_outliers[column], palette="Set2")
                plt.title(f"{title_prefix}: After Outlier Removal - {column}", fontsize=16)

                # Show the plot
                plt.show()

    @staticmethod
    def generate_and_display_all_plots(data, thresholds):
        # Plot Correlation Matrix Heatmap
        DataVisualizer.plot_correlation_heatmap(data)

        # Visualize outliers before and after removal for each specified column
        for column, column_thresholds in thresholds.items():
            DataVisualizer.visualize_outliers_before_after(data, columns=[column], title_prefix="BoxPlot", thresholds=column_thresholds)

        # Create an instance of the LoanDataEDA class
        eda = LoanDataEDA(data)

        loan_status_counts_plot = eda.loan_status_counts_bar_plot()
        loan_status_by_grade_bar_plot = eda.loan_status_by_grade_bar_plot()
        custom_loan_status_bar_plot = eda.custom_loan_status_bar_plot()
        term_loan_amount_violin_plot = eda.term_loan_amount_violin_plot()
        custom_grade_bar_plot = eda.custom_grade_bar_plot()
        loan_status_by_grade_stacked_bar_plot = eda.loan_status_by_grade_stacked_bar_plot()
        categorical_variable_plots = eda.categorical_variable_plots()
        employment_length_group_bar_plot = eda.employment_length_group_bar_plot()


        return (
            loan_status_counts_plot,
            loan_status_by_grade_bar_plot,
            custom_loan_status_bar_plot,
            term_loan_amount_violin_plot,
            custom_grade_bar_plot,
            loan_status_by_grade_stacked_bar_plot,
            categorical_variable_plots,
            employment_length_group_bar_plot
        )

def main():
    # Load data
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

    # Generate and display all plots
    DataVisualizer.generate_and_display_all_plots(data_processor.data, thresholds)

if __name__ == "__main__":
    main()
