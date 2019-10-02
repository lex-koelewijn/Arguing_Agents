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
data = Orange.data.Table(FILE_INPUT)

# Exclude cols: slope, cal and thal which contain a lot of missing values
new_domain = Orange.data.Domain(
    list(data.domain.attributes[:10]),
    list(data.domain.attributes[13:])
)


heart_attack = Orange.data.Table(new_domain, data)
# -

print(heart_attack[0:10])

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

# ### AB part of the algorithm


