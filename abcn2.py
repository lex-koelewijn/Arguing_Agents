# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # ABCN2 Algorithm

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

# ## Plot distributions of classes per parameter

for column in df.columns:
    print(column)
    plt.scatter(df[df['num'] == 0].index, df[df['num'] == 0][column], c = 'g')
    plt.scatter(df[df['num'] == 1].index, df[df['num'] == 1][column], c = 'r')
    plt.figure()
    plt.show()

print(df[df['num'] == 0].mean() < df[df['num'] == 1].mean())
# Only in the case of restecg and thalach the lower the score the lower the probability of being ill

# ## Create Expert Knowledge

# We want to create expert knowledge based rules. Since we do not have the expertise to create these rules ourselves, we need to come up with a way to create rules. The idea of CN2 is that a rule is created and then the data entries in the data that are covered by the rule are removed, and then the process is repeated. The amount of rows that the rule covers says something about the importance of the rule. The more rows are removed the more the rule says about the data, and the more powerfull it is. 
#
# In the end we want to experiment with the percentage of expert knowledge and machine learning knowledge. Therefore we want to create a function that returns a rule given the percentage of rows it removes. Ideally we want this to be a singular rule, since an expert is more likely to come up with a basic rule that covers most of the patients.
#
# Before all this we want to know which parameters have the most influence on the occurrence of a heart attack, which we can later use in the creation of expert knowledge rules.

# +
# Select all the rows in the DataFrame except the last 10,
# which are used for classification
test_rows = int(-0.1* len(df))
df = df.iloc[0:test_rows]
test_df = df.iloc[test_rows:]

print(len(df))
print(len(test_df))

# Dataframe with only labels
df_labels = df['num']

# DataFrame without labels
df_not_num = df[df.columns.difference(['num'])]

df.head()

# +
# NOTE: MAKE SURE P-VALUES AND COLUMN NAMES ARE THE SAME

# Get p-values using ANOVA
p_values = f_classif(df_not_num, df_labels)[1]

# Combine p-values and correct parameters
p_values_dict = {}
for idx, value in enumerate(df_not_num.columns):
    p_values_dict[value] = p_values[idx]
    
# Sort dictionary based on p-values, lowest first
p_values_dict_sorted = sorted(p_values_dict.items(), key=operator.itemgetter(1))
p_values_dict_sorted
# -

print(len(df[(df['age'] >= 54) & (df['num'] == 0)]))
print(len(df[(df['age'] >= 54) & (df['num'] == 1)]))
print(len(df[df['age'] >= 54]))


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
for index, column in enumerate(df_not_num.columns):
    parameter, value, operator, num, new_df = create_rule(df, column, 0.1, 0.5)
    
    rules = {}
    rules[parameter] = (int(value), operator, num)
    rules_dict[index] = rules

return_df = df[~df.apply(tuple,1).isin(new_df.apply(tuple,1))]
print(rules_dict)

# SORT RULES BASED ON ACCURACY 
# -

# ## CN2 Algorithm

# Documentation: https://github.com/biolab/orange3, https://docs.biolab.si//3/data-mining-library/

# +
# Toggle this code to run the CN2 classifier with the last rule found by the rule
# In this case: IF trestbps>=140.0 THEN num=1
return_df.to_csv("data/limited_data.csv", index=None)
data = Orange.data.Table("data/limited_data.csv")

# Only use CN2 run the line below:
# data = Orange.data.Table(FILE_INPUT)

# Find index of column with 'num' in it
num_idx = [idx for idx, value in enumerate(data.domain.attributes) if str(value) == 'num']


# Exclude cols: slope, cal and thal which contain a lot of missing values
new_domain = Orange.data.Domain(
    list(data.domain.attributes[:5] + data.domain.attributes[6:]),
    data.domain.attributes[num_idx[0]]
)

heart_attack = Orange.data.Table(new_domain, data)

# Split data into train and test data 70/30
heart_attack_test = heart_attack[(int(len(heart_attack) * 0.9)):]
heart_attack = heart_attack[:-(int(len(heart_attack) * 0.1))]
# -

heart_attack.domain.attributes

for x in heart_attack.domain.attributes:
    n_miss = sum(1 for d in heart_attack if np.isnan(d[x]))
    print("%4.1f%% %s" % (100.0 * n_miss / len(heart_attack), x.name))

heart_attack.domain.class_var

# +
# Construct a learning algorithm and classifier
cn2_learner = Orange.classification.rules.CN2Learner()
cn2_classifier = cn2_learner(heart_attack)

# Print out the found rules, with the quality of the rule, and curr_class_dist
for rule in cn2_classifier.rule_list:
    print(rule.curr_class_dist.tolist(), rule, rule.quality)
# -

# Print probabilities for both classes per test data entry
cn2_classifier(heart_attack_test, True)

# +
# Iterate over test data and show the classified and actual labels
correctly_classified = 0
for idx, x in enumerate(cn2_classifier(heart_attack_test, True)):
    if x[0] > x[1]:
        if heart_attack_test[idx][10] == 0: correctly_classified += 1
        print('Classified label = 0, actual label ', heart_attack_test[idx][10])
    else:
        if heart_attack_test[idx][10] == 1: correctly_classified += 1
        print('Classified label = 1, actual label ', heart_attack_test[idx][10])
        
"Correctly classified unseen test data out of all unseen test data", correctly_classified, len(heart_attack_test)
# -

# ### Overall performance on train data

# Print amount of correctly classified train data
correctly_classified, incorrectly_classified = 0, 0
for idx, x in enumerate(cn2_classifier(heart_attack, True)):
    if x[0] > x[1]:
        predicted_label = 0
    else:
        predicted_label = 1
    if predicted_label == heart_attack[idx][10]:
        correctly_classified += 1
    else:
        incorrectly_classified += 1
"Correctly and incorrectly classified train data ", correctly_classified, incorrectly_classified


