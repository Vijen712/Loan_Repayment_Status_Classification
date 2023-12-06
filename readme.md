# Predictive Modeling of Loan Repayment Behaviour

## Overview
This project aim is to allow banks to conduct post loan approval classification to determine how the borrowers would repay them before approving the Loan.

## Dataset

https://huggingface.co/datasets/codesignal/lending-club-loan-accepted

## Github Link

https://github.com/Vijen712/Loan_Repayment_Status_Classification

## Requirements

python version : `python-3.10.6`

Package requirements :`pip install -r requirements.txt`

## Deployment

### Steps to run the project:
* Download the zip file and unzip it, the folder structure is Code, Readme, project report
* Download data from the dataset link above and save it in the unziped code folder, which contains all the .py files
* The first step is to run the `pip install -r requirements.txt` in your terminal giving the code folder path (your_path/code).
* To run the EDA and Visualization, run the visualization.py file (your_path/visualization.py)
* The visualization.py file will run the preprocessing steps and display the correlation plot at first, on closing it the next plot popsup. Keep closing the popup figures to view next plots.
* To run the modeling and evaluation, run the main.py file (your_path/main.py)
* After each model displays the ROC curve, when closing the pop up figure, it will show the classification report, and you will get a enter to continue line in your terminal to proceed with the next model. Keep repeating that step till all the models are run.


## File Description 
**main.py** : if main.py is called it does the data preprocessing and data splitting , followed by modeling and evaluation, so main.py calls data_managment , modeling.py

**data_management.py** : Defines a class DataProcessor that encapsulates data preprocessing steps for a loan repayment prediction project. 

**eda.py** : File has the visualizations of exploratoray data analysis plots

**visualization** : if visualization.py is called, it triggers preprocessing and then eda, giving all the eads   data_managment, one part cleaning part pull dtaa from there eda

**modeling.py** : it is incorpated with the model codes, called by main .py

## Contributions

**Business Understanding, Data Collection**- Charan, Vijen, Gyana

**Data Preprocessing** - Charan, Gyana, Vijen

**EDA** - Gyana, Charan, Vijen

**Data Transformation** - Vijen, Gyana , Charan

**Modeling** - Vijen, Charan, Gyana

**Evaluation** - Gyana, Vijen , Charan

**Modular programming, IDE, Github** - Charan, Vijen, Gyana

**PPT and Report** - Charan, Vijen, Gyana







