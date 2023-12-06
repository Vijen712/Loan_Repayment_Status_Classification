import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas

class LoanDataEDA:
    def __init__(self, loan_data):
        self.loan_data = loan_data.copy()

    def loan_status_counts_bar_plot(self):
        loan_status_counts = self.loan_data['loan_status'].value_counts().hvplot.bar(
            title="Loan Status Counts",
            xlabel='Loan Status',
            ylabel='Count',
            width=500,
            
            height=350
        )
        return loan_status_counts


    def loan_status_by_grade_bar_plot(self):
        fully_paid = self.loan_data.loc[self.loan_data['loan_status'] == 'Fully Paid', 'grade'].value_counts().hvplot.bar(
            title="Loan Status by Grade (Fully Paid)",
            xlabel='Grades',
            ylabel='Count',
            width=500,
            height=200
        )
        
        charged_off = self.loan_data.loc[self.loan_data['loan_status'] == 'Charged Off', 'grade'].value_counts().hvplot.bar(
            title="Loan Status by Grade (Charged Off)",
            xlabel='Grades',
            ylabel='Count',
            width=500,
            height=200
        )

        # Combine the two plots
        bar_plot = fully_paid + charged_off

        return bar_plot
    
    def custom_loan_status_bar_plot(self):
        # Assuming loan_data_EDA is your DataFrame
        loan_status_counts = self.loan_data['loan_status'].value_counts()

        # Define a custom color palette
        custom_palette = sns.color_palette("Set3")  # You can choose another palette if you prefer

        # Create a bar chart with different colors for each class
        plt.figure(figsize=(12, 7))
        loan_status_counts.plot(kind='bar', color=custom_palette)
        plt.title("Loan Status Counts")
        plt.xlabel("Loan Status")
        plt.ylabel("Count")
        plt.show()


    def term_loan_amount_violin_plot(self):
        # Set the figure size
        plt.figure(figsize=(12, 7))
        # Plot the violin plot
        sns.set(rc={'figure.figsize': (10, 6)})
        violin_plot = sns.violinplot(x='loan_status', y='loan_amnt', data=self.loan_data, hue='term', split=True, palette='mako')

        # Set plot title and labels
        plt.title("term - loan_amount", fontsize=18)
        plt.xlabel('loan_status', fontsize=12)
        plt.ylabel('loan_amount', fontsize=12)

        # Show the plot
        plt.show()

    
    def custom_grade_bar_plot(self):

        self.loan_data['grade'] = pd.Categorical(self.loan_data['grade'], categories=['A', 'B', 'C', 'D', 'E', 'F', 'G'], ordered=True)

        custom_palette = sns.color_palette("Set2")

        plt.figure(figsize=(12, 7))
        grade_bar_plot = sns.barplot(x='grade', y='loan_amnt', data=self.loan_data, order=['A', 'B', 'C', 'D', 'E', 'F', 'G'], palette=custom_palette)
        plt.title('Loan Amount by Grade')
        plt.show()


        return grade_bar_plot
    
    def loan_status_by_grade_stacked_bar_plot(self):

        fully_paid = self.loan_data.loc[self.loan_data['loan_status'] == 'Fully Paid', 'grade'].value_counts()
        charged_off = self.loan_data.loc[self.loan_data['loan_status'] == 'Charged Off', 'grade'].value_counts()
        late = self.loan_data.loc[self.loan_data['loan_status'] == 'Late', 'grade'].value_counts()
        data = pd.DataFrame({'Fully Paid': fully_paid, 'Charged Off': charged_off, 'Late': late}).fillna(0)
        custom_palette = sns.color_palette("Set3")

        plt.figure(figsize=(12, 7))
        stacked_bar_chart = data.plot(kind='bar', stacked=True, color=custom_palette, width=0.8)
        plt.title("Loan Status by Grade")
        plt.xlabel("Grades")
        plt.ylabel("Count")
        plt.legend(title='Loan Status', loc='upper right')
        plt.xticks(rotation=90)
        plt.show()


        return stacked_bar_chart
    
    def categorical_variable_plots(self):
        # Create subplots
        plt.figure(figsize=(14, 10))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        plt.subplot(4, 2, 1)
        sns.countplot(x='term', data=self.loan_data, hue='loan_status')
        plt.title('Loan Status by Term')

        plt.subplot(4, 2, 2)
        sns.countplot(x='home_ownership', data=self.loan_data, hue='loan_status')
        plt.title('Loan Status by Home Ownership')

        plt.subplot(4, 2, 3)
        sns.countplot(x='verification_status', data=self.loan_data, hue='loan_status')
        plt.title('Loan Status by Verification Status')

        plt.subplot(4, 2, 4)
        g = sns.countplot(x='purpose', data=self.loan_data, hue='loan_status')
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        plt.title('Loan Status by Purpose')

        plt.tight_layout()  # Add this line to improve subplot spacing
        plt.show()

    
    def employment_length_group_bar_plot(self):
        self.loan_data['emp_length_group'] = 'Less than 5 years'
        self.loan_data.loc[self.loan_data['emp_length'].isin(['5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']), 'emp_length_group'] = '5 years or more'
        counts = self.loan_data.groupby(['emp_length_group', 'loan_status']).size().unstack()
        colors = sns.color_palette("Set2")
        fig, ax = plt.subplots(figsize=(12, 7))
        counts.plot(kind='bar', stacked=True, ax=ax, rot=0, color=colors)
        ax.set_xticklabels(['Less than 5 years', '5 years or more'])
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Loan Status based on Employment Length')

        plt.show()


