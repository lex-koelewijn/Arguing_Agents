# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # To-be-named algorithm

# The following notebook is used to import a dataset, and apply the ABCN2 algorithm to it. This uses the CN2 rule based learning algorithm, as well as expert rules which we will also derive from the dataset in this notebook
#
# Data: Heart Attack Prediction, https://www.kaggle.com/imnikhilanand/heart-attack-prediction

# ## Imports and reading data

# +
import Orange
import operator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.2f}'.format
# -

FILE_INPUT = 'data/data_shuffled.csv'

df = pd.read_csv(FILE_INPUT)
df = df.replace('?', np.nan)

# ## Data exploration

df.head()

# Number of NaNs in the dataframe
df.isnull().sum()

# +
# Remove cols with a lot of NaNs
df_with_nan = df[df.columns.difference(['slope', 'ca', 'thal'])]
df = df_with_nan.dropna()

# Print the amount of rows containing NaNs removed
print('Amount of rows containing NaNs removed from original dataset: ', len(df_with_nan) - len(df))
df.head()
# -

print('Amount of patients without and with heart attack: ', len(df[df['num'] == 0]), len(df[df['num'] == 1]))

# Set the object columns to numeric columns
df = df.apply(pd.to_numeric, errors='coerce')
df = df.astype('int64')
df.info()

df.describe()

# Prints unique values
df.nunique()

# ## Creating train/test sets

# +
# Select all the rows in the DataFrame except the last 10 percent,
# which are used for classification
test_percentage = 20
test_rows = int(-(test_percentage/100) * len(df))
train_df = df.iloc[0:test_rows]
test_df = df.iloc[test_rows:]

print(f"train_df len:\t{len(train_df)}")
print(f"test_df len:\t{len(test_df)}")

# Dataframe with only labels
df_labels = df['num']

# DataFrame without labels
train_df_not_num = train_df[train_df.columns.difference(['num'])]
test_df_not_num = test_df[test_df.columns.difference(['num'])]

print(train_df.head())
print(test_df.head())
print(train_df_not_num.head())
print(test_df_not_num.head())


# -

# ## Create Expert Knowledge

# We want to create expert knowledge based rules. Since we do not have the expertise to create these rules ourselves, we need to come up with a way to create rules. The idea of CN2 is that a rule is created and then the data entries in the data that are covered by the rule are removed, and then the process is repeated. The amount of rows that the rule covers says something about the importance of the rule. The more rows are removed the more the rule says about the data, and the more powerfull it is. 
#
# In the end we want to experiment with the percentage of expert knowledge and machine learning knowledge. Therefore we want to create a function that returns a rule given the percentage of rows it removes. Ideally we want this to be a singular rule, since an expert is more likely to come up with a basic rule that covers most of the patients.
#
# Before all this we want to know which parameters have the most influence on the occurrence of a heart attack, which we can later use in the creation of expert knowledge rules.

def create_rule(df, parameter, percentage_expert_knowledge, doubt_percentage):
    # Get length of entire DataFrame and amount of rows that must removed by the rule
    length_df = len(df)
    length_subset_df = int(len(df) * percentage_expert_knowledge)
    
    # Get the minimum and maximum 
    max_value = int(df[parameter].max())
    min_value = int(df[parameter].min())
    
    # For all parameters except 'thalach' and 'restecg' high values indicate higher chance,
    # of heart attack.
#     num_value = False if (parameter == 'thalach' or parameter == 'restecg') else num_value = True
    if parameter == 'thalach' or parameter == 'restecg':
        num_value = 0
#         df = df[df['num'] == 0]
    else: 
        num_value = 1
#         df = df[df['num'] == 1]
    
    # Loop from max to min and find the value where the rule covers 10% of the DataFrame.
    # Values van 1 zijn groter dan die van 0
    value = -1
    new_df = df
    
    iterator = [x/10 for x in range(max_value * 10, min_value * 10, -1)]
    
    for i in iterator:
        new_df = df[(df[parameter] >= i) & (df['num'] == num_value)]
        new_df_all = df[df[parameter] >= i]
#         new_df_negative = df[(df[parameter] >= i) & (df['num'] != num_value)]
        if len(new_df) >= length_subset_df:
            value = i
            break
    
    print('IF ' + str(parameter) + '>=' + str(value) + ' THEN num=' + str(df['num'].values[0]) + '\n' \
           + ' rows covered = ' + str(len(new_df)) + ' of minimal number of rows = ' + str(length_subset_df))
    print('Number of rows where num is opposite of rule: ' + str(len(new_df_all) - len(new_df)) + ' from total ' + (str(len(new_df_all))))
    print('Accuracy is: '  + str(len(new_df) / len(new_df_all) * 100))
    
#     return parameter, value, 1, df['num'].values[0], new_df
    return parameter, value, 1, num_value, new_df


# +
rules_dict = {}
for index, column in enumerate(train_df_not_num.columns):
    parameter, value, operator, num, new_df = create_rule(df, column, 0.1, 0.5)
    
    rules = {}
    rules[parameter] = (int(value), operator, num)
    rules_dict[index] = rules

return_df = df[~df.apply(tuple,1).isin(new_df.apply(tuple,1))]
print(rules_dict)

# SORT RULES BASED ON ACCURACY 
# -

# Documentation: https://github.com/biolab/orange3, https://docs.biolab.si//3/data-mining-library/

# ## CN2 Classification

### Convert datasets to orange format
# train set
train_df.to_csv("data/train_df.csv", index=None)
train_df_orange = Orange.data.Table("data/train_df.csv")
# train set not num
train_df_not_num.to_csv("data/train_df_not_num.csv", index=None)
train_df_not_num_orange = Orange.data.Table("data/train_df_not_num.csv")
# test set not num
test_df_not_num.to_csv("data/test_df_not_num.csv", index=None)
test_df_not_num_orange = Orange.data.Table("data/test_df_not_num.csv")

# +
# Find the index of num in the train set
num_idx = [idx for idx, value in enumerate(train_df_orange.domain.attributes) if str(value) == 'num']
# Create orange domain (lets orange know what the (in)dependent variables are)
orange_domain = Orange.data.Domain(list(train_df_not_num_orange.domain.attributes), train_df_orange.domain.attributes[num_idx[0]])
# Apply this domain to the train and test sets
train_orange_table = Orange.data.Table(orange_domain, train_df_orange)
test_orange_table = Orange.data.Table(orange_domain, test_df_not_num_orange)

print(test_orange_table.domain.attributes)
print(test_orange_table.domain.class_var)

# +
# Create a CN2 instance and train it on the train_orange_table
cn2 = Orange.classification.rules.CN2Learner()
cn2_trained = cn2(train_orange_table)

# # Print the rules it found
# for rule in cn2_trained.rule_list:
#     print(rule)

# Classify the test set
cn2_labels = list(cn2_trained(test_orange_table, False))
actual_labels = list(test_df['num'])
print(f"CN2:\t{cn2_labels}")
print(f"Actual:\t{actual_labels}")

num_correct = 0
for cn2, actual in zip(cn2_labels, actual_labels):
    if cn2 == actual:
        num_correct +=1
print(f"Accuracy:\t{round((num_correct/len(actual_labels))*100, 1)}%")

# +
features = ['age', 'chol', 'cp', 'exang', 'fbs', 'oldpeak', 'restecg', 'sex', 'thalach', 'trestbps']
for i, ftr in enumerate(features):
        if(rules_dict[i][ftr][1] == 1):
            operator = '>='
        else:
            operator = '<'
        print(f"if {ftr} {operator} {rules_dict[i][ftr][0]}, then num={rules_dict[i][ftr][2]}")


    
        
