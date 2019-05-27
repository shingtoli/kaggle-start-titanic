import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import re

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("./input/train.csv")

dataset['Salutation'] = dataset['Name'].apply(lambda x: re.search('[A-Z][a-z]*\.|$', x).group(0)).astype('category')
dataset['Sex'] = dataset['Sex'].astype('category')
dataset['Embarked'] = dataset['Embarked'].fillna('N').astype('category')
ages = dataset.groupby(['Salutation'])['Age'].mean()

print(ages)
dataset['Age'] = dataset.apply(lambda row: ages[row['Salutation']] if np.isnan(row['Age']) else row['Age'], axis=1)

survival = dataset.groupby(['Survived'])
print(survival)

survival_by_gender = dataset.groupby(['Sex', 'Survived']).agg({'Sex': 'count'})
percentages = survival_by_gender.apply(lambda x: 100 * x / float(x.sum()))
print(percentages)


X = dataset.drop(columns=['Name', 'Ticket', 'Cabin']).iloc[:, 2:]
y = dataset.iloc[:, 1]

# Convert categories to numbers
categories = X.select_dtypes(['category']).columns
X[categories] = X[categories].apply(lambda x: x.cat.codes)

selector = SelectKBest(score_func=chi2, k=4)
feat = selector.fit(X, y)

featList = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(feat.scores_)], axis=1)
featList.columns = ['Feature', 'Score']
print(featList.sort_values(by='Score', ascending=False))

# print(feat)

# TODO: Train only on Top 4 Features
# TODO: Run K-fold cross-validation on various models
# TODO: Create final model

pdb.set_trace()
