# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# # ABCN2 Algorithm

# The following notebook is used to import a dataset, and apply the ABCN2 algorithm to it. This uses the CN2 rule based learning algorithm, as well as expert rules which we will also derive from the dataset in this notebook
#
# Data: Heart Attack Prediction, https://www.kaggle.com/imnikhilanand/heart-attack-prediction/downloads/heart-attack-prediction.zip/1

# ## Imports and reading data

import numpy as np
import pandas as pd
import Orange

FILE_INPUT = 'data.csv'

df = pd.read_csv(FILE_INPUT)
df = df.replace('?', np.nan)

# ## Data exploration

df.head()

df.info()

df.describe()

# Prints unique values
df.nunique()

# Number of NaNs in the dataframe
df.isnull().sum()

# ## CN2 Algorithm

# Documentation: https://github.com/biolab/orange3, https://docs.biolab.si//3/data-mining-library/

# +
heart_attack = Orange.data.Table(FILE_INPUT)

# Exclude cols: slope, cal and thal which contain a lot of missing values
# new_domain = Orange.data.Domain(
#     list(data.domain.attributes[:9])
# )

# heart_attack = Orange.data.Table(new_domain, data)
# -

heart_attack.domain.attributes[:9]

heart_attack.domain.class_var

# +
# Construct a learning algorithm and classifier
cn2_learner = Orange.classification.rules.CN2Learner()
cn2_classifier = cn2_learner(heart_attack)

# Print out the found rules, with the quality of the rule, and curr_class_dist
for rule in cn2_classifier.rules:
    print Orange.classification.rules.rule_to_string(rule)
# -

# ### AB part of the algorithm

# +
# Lets create a rule
# IF age>50.0 THEN num=1<0.000, 8.000>

# print Orange.data.filter.Values.Greater
# print Orange.data.filter.ValueFilter.Greater
# print Orange.feature.Descriptor(obj='age')
# print heart_attack.domain.attributes[0]
# print Orange.data.Value(heart_attack.domain.attributes[0], 5)
# print Orange.core.Filter_values.Greater
# print Orange.classification.ConstantClassifier(variable='age', value=5)

# rule = Orange.classification.rules.Rule(filter=Orange.data.filter.ValueFilter.Greater, 
#                                         classifier=Orange.classification.ConstantClassifier(variable='age', value=50), 
#                                         lr=Orange.classification.rules.CN2Learner, 
#                                         dist=Orange.statistics.distribution.Distribution('age', heart_attack), 
#                                         ce=heart_attack)
# print Orange.data.filter.Values.Greater
print Orange.data.filter.Values
print Orange.data.filter.Filter
print Orange.core.ValueFilter.Greater
print Orange.core.Filter


# rule = Orange.classification.rules.Rule(filter=Orange.core.Filter, 
#                                         classifier=Orange.classification.ConstantClassifier(Orange.data.Value(heart_attack.domain.attributes[0], 5)),
#                                         lr=Orange.classification.rules.CN2Learner, 
#                                         dist=Orange.statistics.distribution.Distribution('num', heart_attack), 
#                                         ce=heart_attack)

rule = Orange.classification.rules.Rule(filter=Orange.data.filter.Filter,
                                        classifier=Orange.classification.ConstantClassifier(Orange.data.Value(heart_attack.domain.attributes[0], 50)))
#                                         lr=Orange.classification.rules.CN2Learner, 
#                                         dist=Orange.statistics.distribution.Distribution('num', heart_attack), 
#                                         ce=heart_attack)

print(Orange.classification.rules.rule_to_string(rule))
rule_list = Orange.classification.rules.RuleList()

# Print out the found rules, with the quality of the rule, and curr_class_dist
print(rule_list)

# class Orange.classification.rules.RuleCovererAndRemover
# Base class for rule coverers and removers that, when invoked, remove instances covered by the rule and return remaining instances.
# Orange.classification.rules.ABCN2(rules=,
#                                  instances=heart_attack)
# -


