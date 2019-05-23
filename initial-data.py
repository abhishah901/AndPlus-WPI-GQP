import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('dataset_full.csv')

# print("Inside RF:")
# print(data.ix[:, -3].head())
# y = data.ix[:, -3]
# X = data.ix[:]
N = data.size
print(data.head())
print(data.describe())
def correlation(data):
    corr = data.corr()
    sns.heatmap(corr, annot=True)
    # plt.savefig('7k.png')
    return

# neg = data.loc[data['TimeToStart'] < 0]
# print("Neg: ",neg.count())

# correlation(data)
colormap  = {True : 'Blue', False : 'Red'}
holder = data['InNormalRange']
print(holder.head())
# holder = holder.map(colormap)
# # print(holder.head())
# print(holder.describe())

def visualize(x, y, c = holder):
    # print(x.head())
    # print(y.head())
    # normal_x = (x - x.mean()) / (x.max() - x.min())
    # normal_y = (y - y.mean()) / (y.max() - y.min())
    # print(normal_x.head())
    # print(normal_y.head())
    # print(normal_y.max(),normal_y.min())
    # print(normal_x.max(),normal_x.min())
    plt.scatter(x = x, y = y, c = c)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # print(type(x))
    # print(type(y))
    # plt.show()
    return

# temper = zip(data['ContentLength'],data['TimeSpent'])


def client():
    # print(data.groupby('InNormalRange').count()['ID'])
    clients = data.groupby('Group').count()['InNormalRange']
    # print("Description: \n",clients.describe())
    print("Top 5 clients with most tickets:\n",clients.sort_values(ascending = False).head(5))
    # print(clients.head())
    # print(clients.columns.values)
    # fg = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
    # fg.map(pyplot.scatter, 'Weight (kg)', 'Height (cm)').add_legend()
    print("Total number of clients: ",len(clients))
    plt.hist(x=clients, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    # xticks = (np.arange(clients))
    # plt.show()


def RF(ne, md):
    clf = RandomForestClassifier(n_estimators = ne, max_depth = md, random_state = 0)
    print("Inside RF:")
    # print(data.ix[:,-3].head())
    y = data.ix[:,-3]
    X = data.drop(['InNormalRange', 'Industry', 'RequestMonth', 'Group'], axis = 1)
    # print(X.columns.values)
    clf.fit(X, y)
    # imp = list(zip(X.columns.values, clf.feature_importances_))
    imp = pd.DataFrame(clf.feature_importances_,index = X.columns.values)
    print(imp)
    imp.plot(kind='barh')
    # plt.show()
    # features = X.column.values
    # importances = clf.feature_importances_
    # indices = np.argsort()
    # f = pd.DataFrame()
    # print(clf.best_estimator_.features_importances_)
    # print(clf.predict)
    # print(clf.predict([[0, 0, 0, 0]]))
    # print(y.head())
    # print(X.head())
    # print(len(y))
    # print(data.columns.values)
    # print(X.columns.values)


RF(100, 2)
client()
visualize(data['ContentLength'],data['TimeSpent'])
visualize(data['ContentLength'],data['TimeToStart'])
visualize(data['TimeSpent'],data['TimeToStart'])
# plt.show()
# print(data['Score'].max())