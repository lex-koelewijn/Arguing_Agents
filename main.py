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

# # expCN2 Algorithm

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

FILE_INPUT = 'data/data.csv'

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
# which are used for classification, we can do this because the data
# is shuffled.
# Select all the rows in the DataFrame except the last n percent,
# which are used for classification

test_percentage = 15
test_rows = int(-(test_percentage/100) * len(df))
# Static Sampling
#train_df = df.iloc[0:test_rows]
#test_df = df.iloc[test_rows:]
# Random Sampling
train_df = df.sample(int((1-(test_percentage/100)) * len(df)), random_state=421)
test_df = df[~df.apply(tuple,1).isin(train_df.apply(tuple,1))].sample(frac=1, random_state=421)
# DataFrame without labels
train_df_not_num = train_df[train_df.columns.difference(['num'])]
test_df_not_num = test_df[test_df.columns.difference(['num'])]

print(train_df.head())
print(test_df.head())
print(train_df_not_num.head())
print(test_df_not_num.head())

# print(train_df.head())
# print(test_df.head())
# print(train_df_not_num.head())
# print(test_df_not_num.head())

# print(f"Rows where oldpeak>=2.0 (train/test):\t({len(train_df.loc[train_df['oldpeak'] >= 2.0])}/{len(test_df.loc[test_df['oldpeak'] >= 2.0])})")
# print(f"Rows where thalach>=170 (train/test):\t({len(train_df.loc[train_df['thalach'] >= 170])}/{len(test_df.loc[test_df['thalach'] >= 170])})")

print(f"df total length:\t{len(df)}")
print(f"train_df total length:\t{len(train_df)}")
print(f"test_df total length:\t{len(test_df)}")


# -

# ## Create Expert Knowledge

# We want to create expert knowledge based rules. Since we do not have the expertise to create these rules ourselves, we need to come up with a way to create rules. The idea of CN2 is that a rule is created and then the data entries in the data that are covered by the rule are removed, and then the process is repeated. The amount of rows that the rule covers says something about the importance of the rule. The more rows are removed the more the rule says about the data, and the more powerfull it is. 
#
# In the end we want to experiment with the percentage of expert knowledge and machine learning knowledge. Therefore we want to create a function that returns a rule given the percentage of rows it removes. Ideally we want this to be a singular rule, since an expert is more likely to come up with a basic rule that covers most of the patients.
#
# Before all this we want to know which parameters have the most influence on the occurrence of a heart attack, which we can later use in the creation of expert knowledge rules.

# Print for which variables higher values indicate heart attack,
# using the mean.
print(df[df['num'] == 0].mean() < df[df['num'] == 1].mean())

# +
# Use ANOVA to find the parameters which have the highest influence,
# on the target value
df_labels = df['num']
df_not_num = df[df.columns.difference(['num'])]

# Get p-values using ANOVA
p_values = f_classif(df_not_num, df_labels)[1]

# Combine p-values and correct parameters
p_values_dict = {}
for idx, value in enumerate(df_not_num.columns):
    p_values_dict[value] = p_values[idx]
    
# # Sort dictionary based on p-values, lowest first
p_values_dict_sorted = sorted(p_values_dict.items(), key=operator.itemgetter(1))
p_values_dict_sorted


# -

def create_rule(df, parameter, percentage_expert_knowledge, high_to_low):
    # Get length of entire DataFrame and amount of rows that must removed by the rule
    length_df = len(df)
    length_subset_df = int(len(df) * percentage_expert_knowledge)
    
    # Get the minimum and maximum of given parameter
    max_value = int(df[parameter].max())
    min_value = int(df[parameter].min())
    
    # For all parameters except 'thalach' and 'restecg' high values indicate higher chance,
    # of heart attack.
    if parameter == 'thalach' or parameter == 'restecg':
        num_value = (1 - high_to_low)
    else: 
        num_value = high_to_low
    
    # Loop from max to min and find the value where the rule given percentage of DataFrame.
    # Initialize values
    value = -1
    acc_score = 0.0
    new_df = df
    
    # Create an iterator that goes from max to min in intervals of 0.1
    if(high_to_low == 1):
        iterator = [x/10 for x in range(max_value * 10, min_value * 10, -1)]
        operator_str = '>='
        operator = 1
    else:
        iterator = [x/10 for x in range(min_value * 10, max_value * 10,  1)]
        operator_str = '<'
        operator = 0
        
    for i in iterator:
        if(high_to_low == 1):
            new_df = df[df[parameter] >= i]
        else:
            new_df = df[df[parameter] < i]
        if len(new_df) >= length_subset_df:
            value = i
            acc_score = len(new_df[new_df['num'] == num_value]) / len(new_df)
            break
            
    
    
    print('IF ' + str(parameter) + operator_str + str(value) + ' THEN num=' + str(num_value) + '\n' \
           + ' rows covered = ' + str(len(new_df)) + ' of minimal number of rows = ' + str(length_subset_df) + '\n' \
           + ' percentage: ' + str(int(len(new_df) / len(df) * 100)) + ' of ' + str(int(percentage_expert_knowledge * 100)) + ', acc_score: ' + str(acc_score))
#     print('Number of rows where num is opposite of rule: ' + str(len(new_df_all) - len(new_df)) + ' from total ' + (str(len(new_df_all))))
#     print('Accuracy is: '  + str(len(new_df) / len(new_df_all) * 100))
    
    return parameter, value, operator, num_value, new_df

# +
rules_dict = {}
for index, (column, p_value) in enumerate(p_values_dict_sorted):
    parameter, value, operator, num, new_df = create_rule(df=df, 
                                                          parameter=column,
                                                          percentage_expert_knowledge=0.1,
                                                          high_to_low=0) #Look at lower end of data [=0] or upper end of data [=1]
    rules = {}
    rules[column] = (int(value), operator, num)
    rules_dict[index] = rules

# NOTE: Need to do this for the correct rule in the rules_dict, now it takes the last dict
train_partial_df = train_df[~train_df.apply(tuple,1).isin(new_df.apply(tuple,1))]
# -

# Documentation: https://github.com/biolab/orange3, https://docs.biolab.si//3/data-mining-library/

# ## CN2 Classifications

# +
### Convert datasets to orange format
# train set
train_df.to_csv("data/train_df.csv", index=None)
train_df_orange = Orange.data.Table("data/train_df.csv")
# train set not num
train_df_not_num.to_csv("data/train_df_not_num.csv", index=None)
train_df_not_num_orange = Orange.data.Table("data/train_df_not_num.csv")
# partial train set
train_partial_df.to_csv("data/train_partial_df.csv", index=None)
train_partial_df_orange = Orange.data.Table("data/train_partial_df.csv")
# test set not num
test_df_not_num.to_csv("data/test_df_not_num.csv", index=None)
test_df_not_num_orange = Orange.data.Table("data/test_df_not_num.csv")

# print(len(train_df_not_num))
# print(len(train_partial_df_not_num))
# -

# Find the index of num in the train set
num_idx = [idx for idx, value in enumerate(train_df_orange.domain.attributes) if str(value) == 'num']
# Create orange domain (lets orange know what the (in)dependent variables are)
orange_domain = Orange.data.Domain(list(train_df_not_num_orange.domain.attributes), train_df_orange.domain.attributes[num_idx[0]])
# Apply this domain to the train and test sets
train_orange_table = Orange.data.Table(orange_domain, train_df_orange)
train_partial_orange_table = Orange.data.Table(orange_domain, train_partial_df_orange)
test_orange_table = Orange.data.Table(orange_domain, test_df_not_num_orange)
# Print to verify
print(train_orange_table.domain)
print(train_partial_orange_table.domain)

# +
# Create CN2 instances and train them
cn2_full = Orange.classification.rules.CN2Learner()
cn2_full_trained = cn2_full(train_orange_table)
cn2_partial = Orange.classification.rules.CN2Learner()
cn2_partial_trained = cn2_partial(train_partial_orange_table)

# Classify the test sets
cn2_full_labels = list(cn2_full_trained(test_orange_table, False))
cn2_partial_labels = list(cn2_partial_trained(test_orange_table, False))

# Print rules of CN2
# for rule in cn2_full_trained.rule_list:
#     print(rule.curr_class_dist.tolist(), rule, rule.quality)
# print("---")
# for rule in cn2_partial_trained.rule_list:
#     print(rule.curr_class_dist.tolist(), rule, rule.quality)
# -

# ## Expert Classification

# +
#Classify the test set according to our expert rules. You go through each measurement of a row in the test set until you find an applicable expert 
#rule and let that rule decide the num that it shoud be classified as. 

#Get the parameter list in the correct order
# parameters = []
# for tup in p_values_dict_sorted:
#     parameters.append(tup[0])

#Please note that the parameter list should contains all names of the parameters used in the rule dictionary in the same order.
# 4 Good Rules
rules_dict = {0: {"cp": (2.1, 0, 0)}, 1: {"oldpeak": (2.0, 1, 1), 2: {'sex': (0.1, 0, 0)}, 3: {"thalach": (170, 1, 0)}}}
parameters = ["cp", "oldpeak", "sex", "thalach"]

#The best rule
# rules_dict = {0: {"cp": (2.1, 0, 0)}}
# parameters = ["cp"]

# 4 bad Rules
# rules_dict = {0: {"restecg": (0.1, 0, 1)}, 1: {"sex": (1.0, 1, 1), 2: {'age': (57, 1, 1)}, 3: {"trestbps": (160, 1, 1)}}}
# parameters = ["restecg", "sex", "sex", "trestbps"]

#The worst rule
# rules_dict = {0: {"restecg": (0.1, 0, 1)}}
# parameters = ["restecg"]


#These are the names of the column in the data table
table_columns = ["age", "chol", "cp", "exang", "fbs", "oldpeak", "restecg", "sex", "thalach", "trestbps"] 
expert_labels = [2]*len(test_df_not_num_orange)
                # 2 looks better than -1 when printing the results

#Loop through test set
i = 0 
for row in test_df_not_num_orange:
    m = 0 
    for rule in rules_dict: 
        if( m > len(parameters)-1):
            break
        #Get the apprioprate value from the current row
        measurement = row[table_columns.index(parameters[m])]
        #Get the rule tuple that is relevant
        rule_tuple = rules_dict[rule][parameters[m]]
        if(rule_tuple[1]==0):
            #measurement must be smaller than
            if(measurement < rule_tuple[0]):
                num_for_measurement = rule_tuple[2]
                #Add the classification of this row by the expert rule to the classification list. 
                expert_labels[i] = num_for_measurement
                break
            else: 
                m += 1     #No applicable expert rule found, try the next measurement
                continue
        elif(rule_tuple[1]==1):
            #measurement must be larger or equal then
            if(measurement >= rule_tuple[0]):
                num_for_measurement = rule_tuple[2]
                #Add the classification of this row by the expert rule to the classification list. 
                expert_labels[i] = num_for_measurement
                break
            else:
                m += 1     #No applicable expert rule found, try the next measurement
                continue
    i += 1
# -
# ## Combining the CN2 and Expert labels


# +
# Checks if expert_labels has a useful label (0 or 1) and copies that,
# if there is not useful label, the cn2 label is copied instead.
def combine_cn2_expert(cn2_labels, expert_labels):
    final_labels = []
    for cn2_label, expert_label in zip(cn2_labels, expert_labels):
        if expert_label == 0 or expert_label == 1:
            final_labels.append(expert_label)
        else:
            final_labels.append(cn2_label)
    return final_labels

final_full_labels = combine_cn2_expert(cn2_full_labels, expert_labels)
final_partial_labels = combine_cn2_expert(cn2_partial_labels, expert_labels)


# -

# ## Calculating scores

# +
# Returns the percentage of correct labels
# If the pred is 2 (expert knowledge), it is not counted as incorrect
def get_accuracy(pred_labels, correct_labels):
    num_correct, size_limiter = (0, 0)
    for pred, correct in zip(pred_labels, correct_labels):
        if pred == 2: 
            size_limiter += 1
        elif pred == correct:
            num_correct +=1
    return round(((num_correct/(len(correct_labels) - size_limiter))*100), 1)
# Returns the precision&recall&f1 given the predicted and correct labels
def get_prf1(pred_labels, correct_labels):
    true_pos, false_pos, true_neg, false_neg = (0, 0, 0, 0)
    precision, recall, f1 = (-1, -1, -1)
    for pred, correct in zip(pred_labels, correct_labels):
        if (pred == 1) and (correct == 1): true_pos += 1
        if (pred == 1) and (correct == 0): false_pos += 1
        if (pred == 0) and (correct == 0): true_neg += 1
        if (pred == 0) and (correct == 1): false_neg += 1
    # Calculate precision
    if true_pos != 0 and false_pos != 0: precision = round(true_pos/(true_pos + false_pos), 2)
    # Calculate recall
    if true_pos != 0 and false_neg != 0: recall = round(true_pos/(true_pos + false_neg), 2)
    # Calculate f1-score
    if precision != -1 and recall != -1: f1 = round(2*((precision * recall) / (precision + recall)), 2)
    return precision, recall, f1
# Returns the percentage of useful labels of a label
def get_coverage(labels):
    num_covered = 0
    for label in labels:
        if label == 0 or label == 1: num_covered +=1
    return round(((num_covered/len(labels))*100), 1)

# The actual labels to compare against
actual_labels = list(test_df['num'])

# +
# Print stuff
print("EXPERT CLASSIFICATION:")
print(f"Expert:\t{expert_labels}")
print(f"Actual:\t{actual_labels}")
print(f"Expert Accuracy:\t{get_accuracy(expert_labels, actual_labels)}%\t({get_coverage(expert_labels)}% coverage)")

print("\nCN2 CLASSIFICATION (FULL TRAIN SET):")
print(f"CN2:\t{cn2_full_labels}")
print(f"Final:\t{final_full_labels}")
print(f"Actual:\t{actual_labels}")
print(f"CN2 Accuracy:\t\t{get_accuracy(cn2_full_labels, actual_labels)}%")
print(f"Precision/Recall/F1:\t{get_prf1(cn2_full_labels, actual_labels)}")
print(f"expCN2 Accuracy:\t{get_accuracy(final_full_labels, actual_labels)}%")
print(f"Precision/Recall/F1:\t{get_prf1(final_full_labels, actual_labels)}")

print("\nCN2 CLASSIFICATION (PARTIAL TRAIN SET):")
print(f"CN2:\t{cn2_partial_labels}")
print(f"Final:\t{final_partial_labels}")
print(f"Actual:\t{actual_labels}")
print(f"CN2 Accuracy:\t\t{get_accuracy(cn2_partial_labels, actual_labels)}%")
print(f"Precision/Recall/F1:\t{get_prf1(cn2_partial_labels, actual_labels)}")
print(f"expCN2 Accuracy:\t{get_accuracy(final_partial_labels, actual_labels)}%")
print(f"Precision/Recall/F1:\t{get_prf1(final_partial_labels, actual_labels)}")

print("\nOTHER INFO:")
print(f"Full train set size:\t{len(train_df)}")
print(f"Partial train set size:\t{len(train_partial_df)}")
print(f"Test set size:\t\t{len(test_df)}")
# -



