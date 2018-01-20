# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import seaborn as sns

def check_discrete(df, col, label='Loan Status', label_key='Charged Off'):
    unq = df[col].unique()
    print(df[col].describe())
    print('Missing: ' + str(df[col].isnull().sum()))
    if len(unq) < 20:
        print(df[col].value_counts())
        print(pd.crosstab(df[col], df[label]).sort_values(label_key))
        print(pd.crosstab(df[col], df[label], normalize='index').sort_values(label_key))
    else:
        print('Unique vals: ' + str(len(unq)))
    return()

def check_cont(df, col, label='Loan Status'):
    print(df[col].describe())
    print('Missing: ' + str(df[col].isnull().sum()))
    df[col].hist(by=df[label], bins=10)
    return()
    
def current_job(x):
    if x == 'n/a':
        return('none')
    elif x == '10+ years':
        return('10+')
    elif x in ['9 years', '8 years', '7 years', '6 years', '5 years']:
        return('long')
    else:
        return('short')

def mort(x):
    if x not in ['Home Mortgage', 'Rent', 'Own Home']:
        return('Home Mortgage')
    else:
        return(x)
        
def purpose(x):
    if x in ['Business Loan', 'small_business', 'renewable_energy', 'moving']:
        return('high')
    elif x in ['vacation', 'Medical Bills']:
        return('mid_high')
    elif x in ['Home Improvements', 'wedding']:
        return('mid_low')
    elif x in ['Educational Expenses', 'Buy a Car']:
        return('low')
    else:
        return('mid')

def credit_probs(x):
    if float(x) < 1:
        return('none')
    elif float(x) < 6:
        return('some')
    else:
        return('many')
    
path = r'.'
data = 'loans.csv'

data = pd.read_csv(os.path.join(path, data))
#data = data.drop_duplicates()
#data = data.dropna().drop_duplicates()

#  Loan ID: A unique Identifier for the loan information.
# Customer ID: A unique identifier for the customer. Customers may have more than one loan.
data = data.drop_duplicates('Loan ID')

# Loan Status: A categorical variable indicating if the loan was paid back or defaulted.

# Current Loan Amount: This is the loan amount that was either completely paid off, or the amount that was defaulted.
#check_cont(data, 'Current Loan Amount')
data['Current Loan Amount'][data['Current Loan Amount'] > 1e6] = np.nan
#sns.violinplot(data['Current Loan Amount'], data['Loan Status'], hue=data['Term'], split=True)
#data.drop('Current Loan Amount', axis=1, inplace=True) # little difference
#data['Current Loan Amount'].fillna(data['Current Loan Amount'].mean(), inplace=True) # bad

# Term: A categorical variable indicating if it is a short term or long term loan.
#check_discrete(data, 'Term') # important

# Credit Score: A value between 0 and 800 indicating the riskiness of the borrowers credit history.
data['Credit Score'] = pd.to_numeric(data['Credit Score'], errors='coerce')
data['Credit Score'] = data['Credit Score'].apply(lambda x: x if np.isnan(x) or x<800 else x/10)
#check_cont(data,'Credit Score')
#sns.violinplot(data['Credit Score'], data['Loan Status'], hue=data['Term'], split=True)

# Years in current job: A categorical variable indicating how many years the customer has been in their current job.
#sns.countplot(data['Years in current job'], hue=data['Loan Status'])
data['Years in current job'] = data['Years in current job'].apply(current_job)
#check_discrete(data,'Years in current job')

# Home Ownership: Categorical variable indicating home ownership. Values are "Rent", "Home Mortgage", and "Own". If the value is OWN, then the customer is a home owner with no mortgage
data['Home Ownership'] = data['Home Ownership'].apply(mort)
#check_discrete(data, 'Home Ownership')

# Annual Income: The customer's annual income
data['Annual Income'] = np.log(data['Annual Income'])
#data['Annual Income'].fillna(data['Annual Income'].mean(), inplace=True) # maybe ok
#check_cont(data, 'Annual Income')

# Purpose: A description of the purpose of the loan.
data['Purpose'] = data['Purpose'].apply(purpose)
#check_discrete(data, 'Purpose')

# Monthly Debt: The customer's monthly payment for their existing loans
data['Monthly Debt'] = pd.to_numeric(data['Monthly Debt'], errors='coerce')
data['Monthly Debt'] = np.log(data['Monthly Debt'] + 1)
data['Monthly Debt'].fillna(data['Monthly Debt'].mean(), inplace=True)
#check_cont(data, 'Monthly Debt')

# Years of Credit History: The years since the first entry in the customer’s credit history
#sns.violinplot(data['Years of Credit History'], data['Loan Status'], hue=data['Term'], split=True)
data['Years of Credit History'] = pd.to_numeric(data['Years of Credit History'], errors='coerce')
data['Years of Credit History'] = np.log(data['Years of Credit History'] + 1)
#check_cont(data, 'Years of Credit History')

# Months since last delinquent: Months since the last loan delinquent payment
data['Months since last delinquent'] = pd.to_numeric(data['Months since last delinquent'], errors='coerce')
data['Months since last delinquent'] = data['Months since last delinquent'].clip_upper(90)
#check_cont(data, 'Months since last delinquent')

# Number of Open Accounts: The total number of open credit cards
data['Number of Open Accounts'] = np.log(data['Number of Open Accounts'] + 1)
#check_cont(data, 'Number of Open Accounts')

# Number of Credit Problems: The number of credit problems in the customer records.
data['Number of Credit Problems'] = data['Number of Credit Problems'].apply(credit_probs)
#check_discrete(data, 'Number of Credit Problems')

# Current Credit Balance: The current total debt for the customer
data['Current Credit Balance'] = data['Current Credit Balance'].clip_upper(2e5)
data['Current Credit Balance'] = np.log(data['Current Credit Balance'] + 1)
#check_cont(data, 'Current Credit Balance')

# Maximum Open Credit: The maximum credit limit for all credit sources.
data['Maximum Open Credit'] = pd.to_numeric(data['Maximum Open Credit'], errors='coerce')
data['Maximum Open Credit'] = data['Maximum Open Credit'].clip_upper(1e6)
data['Maximum Open Credit'].fillna(data['Maximum Open Credit'].mean(), inplace=True)
data['Maximum Open Credit'] = np.log(data['Maximum Open Credit'] + 1)
#check_cont(data, 'Maximum Open Credit')

# Bankruptcies: The number of bankruptcies
data['Bankruptcies'] = pd.to_numeric(data['Bankruptcies'], errors='coerce')
data['Bankruptcies'] = data['Bankruptcies'].apply(lambda x: 'none' if x < 1
else 'few' if x < 4 else 'many')
#check_discrete(data, 'Bankruptcies')

# Tax Liens: The number of tax liens.
data['Tax Liens'] = pd.to_numeric(data['Tax Liens'], errors='coerce')
data['Tax Liens'] = data['Tax Liens'].apply(lambda x: 'none' if x < 1
else 'few' if x < 3 else 'many')
#check_discrete(data, 'Tax Liens')

# MICE: Current Loan Amount, Credit Score, Annual Income, Months since last delinquent
sns.pairplot(data.dropna(), hue='Loan Status', kind='reg')