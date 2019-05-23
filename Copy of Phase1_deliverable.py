# importer

import pickle
from joblib import load
import pandas as pd

# .joblib loader

clf = load('filename.joblib')

# getter

data = pd.read_csv('df_encoded.csv')  # path to csv

# one-hot encoder

df_processed = pd.get_dummies(data, columns=['Industry', 'RequestMonth', 'CompleteMonth', 'ApplicationType','TaskDomain'])

df_processed['HasUserStory_binary'] = df_processed['HasUserStory'] * 1
df_processed['InNormalRange_binary'] = df_processed['InNormalRange'] * 1

# prepper

X = df_processed.drop(['ID', 'ClientId', 'Group'], axis = 1)

# sizer

print("Predictions:", data.size)

# predictor

predicted = clf.predict(X)

# exporter

print(predicted)