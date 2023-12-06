# data_management.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class DataProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def read_data(self):
        # Step 1: Read the CSV file
        self.data = pd.read_csv(self.file_name)

    def preprocess_data(self):
        # Step 2: Process loan status
        self.process_loan_status()
        # Step 3: Remove columns with NaN values
        self.remove_null_values()
        # Step 4: Drop specified columns
        self.drop_columns()
        # Step 5: Drop rows with null values
        self.drop_null_rows()
        # Step 6: Calculate FICO score average
        self.calculate_fico_score_avg()
        # Step 7: Remove high correlation numeric columns
        self.remove_high_corr_numeric_columns()
        # Step 8: Remove high correlation categorical columns
        self.remove_high_corr_categorical_columns()

        self.column_changes()

        # This function will be updated as more preprocessing steps are added
        pass

    def split_data(self):
        # This function will be updated as more splitting logic is added
        pass

    def process_loan_status(self):
        # Step 2.1: Drop rows with specified loan statuses
        self.data = self.data[~self.data['loan_status'].isin(['Current', 'Default'])]

        # Step 2.2: Rename loan statuses
        self.data['loan_status'] = self.data['loan_status'].replace({
            'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid',
            'Does not meet the credit policy. Status:Charged Off': 'Charged Off',
            'Late (31-120 days)': 'Late',
            'Late (16-30 days)': 'Late',
            'In Grace Period': 'Late'
        })

    def remove_null_values(self, threshold=50):
        # Step 3.1: Count the number of NaN values in each column
        null_counts = self.data.isnull().sum()

        # Step 3.2: Calculate the percentage of NaN values in each column
        null_percentage = (null_counts / len(self.data)) * 100

        # Step 3.3: Get the column names that exceed the threshold
        columns_to_drop = null_percentage[null_percentage > threshold].index

        # Step 3.4: Drop the columns from the DataFrame
        self.data = self.data.drop(columns=columns_to_drop)

        # Display the DataFrame after dropping columns
        #print(self.data)

        # # Verify the changes
        # print(self.data['loan_status'].value_counts())

    def drop_columns(self, columns_to_drop=None):
        # Step 4.1: Specify the columns to drop if not provided
        if columns_to_drop is None:
            columns_to_drop = ['funded_amnt', 'funded_amnt_inv', 'emp_title', 'issue_d', 'url','title', 'zip_code','inq_last_6mths', 'revol_bal','total_acc',
                   'initial_list_status', 'out_prncp', 'out_prncp_inv','total_pymnt_inv','total_rec_prncp', 'total_rec_int','total_rec_late_fee', 'recoveries',
                   'collection_recovery_fee','last_pymnt_d', 'last_pymnt_amnt','last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low',
                   'collections_12_mths_ex_med', 'policy_code','tot_cur_bal','total_rev_hi_lim','acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
                   'delinq_amnt', 'mo_sin_old_il_acct','mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl','mths_since_recent_bc', 'mths_since_recent_inq',
                   'num_accts_ever_120_pd','num_il_tl', 'num_op_rev_tl','num_rev_accts', 'num_rev_tl_bal_gt_0','num_tl_120dpd_2m','num_tl_30dpd', 'num_tl_90g_dpd_24m', 
                   'num_tl_op_past_12m','percent_bc_gt_75','tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit','total_il_high_credit_limit','disbursement_method','addr_state','id']

        # Step 4.2: Drop the specified columns from the DataFrame
        self.data = self.data.drop(columns=columns_to_drop)

        # Display the DataFrame after dropping columns
        #print(self.data)
    def drop_null_rows(self):
        # Step 5.1: Drop rows with null values
        self.data = self.data.dropna()
    
    def calculate_fico_score_avg(self):
        # Step 6.1: Create a new column 'fico_score_avg' with the average of 'fico_range_low' and 'fico_range_high'
        self.data['fico_score_avg'] = self.data[['fico_range_low', 'fico_range_high']].mean(axis=1)

        # Step 6.2: Drop the original columns 'fico_range_low' and 'fico_range_high'
        self.data = self.data.drop(['fico_range_low', 'fico_range_high'], axis=1)

        # Display the DataFrame after calculating FICO score average
        # print(self.data)

    def remove_high_corr_numeric_columns(self, threshold=0.8):
        # Step 7.1: Get numeric features
        num_feat = self.data.select_dtypes('number').columns.values

        # Step 7.2: Generate combinations of numeric features
        comb_num_feat = np.array(list(combinations(num_feat, 2)))

        # Step 7.3: Calculate correlations and identify highly correlated pairs
        corr_num_feat = np.array([])
        for comb in comb_num_feat:
            corr = pearsonr(self.data[comb[0]], self.data[comb[1]])[0]
            corr_num_feat = np.append(corr_num_feat, corr)
        high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= threshold]

        # Step 7.4: Drop columns with high correlation
        self.data = self.data.drop(np.unique(high_corr_num[:, 1]), axis=1, errors='ignore')

    def remove_high_corr_categorical_columns(self, threshold=0.8):
        # Step 8.1: Get categorical features
        cat_feat = self.data.select_dtypes('object').columns.values

        # Step 8.2: Generate combinations of categorical features
        comb_cat_feat = np.array(list(combinations(cat_feat, 2)))

        # Step 8.3: Calculate correlations and identify highly correlated pairs
        corr_cat_feat = np.array([])
        for comb in comb_cat_feat:
            table = pd.pivot_table(self.data, values='loan_amnt', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
            corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1)))
            corr_cat_feat = np.append(corr_cat_feat, corr)
        high_corr_cat = comb_cat_feat[corr_cat_feat >= threshold]

        # Step 8.4: Drop columns with high correlation
        self.data = self.data.drop(np.unique(high_corr_cat[:, 1]), axis=1, errors='ignore')
        print(self.data)
    
    def column_changes(self):
        # Additional Transformation 1: Convert the "earliest_cr_line" column to datetime and extract the year
        self.data['earliest_cr_line'] = pd.to_datetime(self.data['earliest_cr_line'], format='%b-%Y')
        self.data['earliest_cr_line'] = self.data['earliest_cr_line'].dt.year

        # Additional Transformation 2: Replace 'ANY' and 'NONE' with 'OTHER' in 'home_ownership'
        self.data['home_ownership'].replace(['ANY', 'NONE'], 'OTHER', inplace=True)
    
    def apply_onehot_encoding(self):

        # Assuming 'df' is your DataFrame
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns

        # Remove 'loan_status' from categorical_cols
        categorical_cols = categorical_cols.drop('loan_status', errors='ignore')

        # Apply one-hot encoding
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)

        # Additional Transformation: Encoding 'loan_status'
        class_mapping = {'Charged Off': 0, 'Fully Paid': 1, 'Late': 2}
        self.data['loan_status_encoded'] = self.data['loan_status'].map(class_mapping)
        self.data = self.data.drop('loan_status', axis=1)
        self.data.rename(columns={'loan_status_encoded': 'loan_status'}, inplace=True)
    
    def split_data(self, test_size=0.15, validation_size=0.15, random_state=42):
        # Step 9: Split the data into features and target
        self.X = self.data.drop('loan_status', axis=1)
        self.y = self.data['loan_status']
        self.X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '') for col in self.X.columns]

        # Step 10: Splitting the data into train, validation, and test sets
        remaining_size = 1 - test_size
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_temp, self.y_temp, test_size=validation_size/remaining_size, stratify=self.y_temp, random_state=random_state
        )


def remove_outliers(df, column, lower_threshold=None, upper_threshold=None, drop_outliers=False):
    if lower_threshold is not None and upper_threshold is not None:
        outliers = df[(df[column] < lower_threshold) | (df[column] > upper_threshold)]
    elif lower_threshold is not None:
        outliers = df[df[column] < lower_threshold]
    elif upper_threshold is not None:
        outliers = df[df[column] > upper_threshold]

    if drop_outliers:
        df = df.drop(outliers.index, axis=0)

    return df, len(outliers)
    