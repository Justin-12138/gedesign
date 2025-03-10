import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from feature_engine.selection import MRMR
# dataset show
data = pd.read_csv(r'/home/lz/Desktop/gdesign/datasets/example_data.csv')

print(data.shape)
print(len(data['Label']))
features = list(data.head(1))
labels = list(data.iloc[:,-1])
data = data.iloc[:, :-1]

print(list(labels))
print(list(features))
f1 = list(data.iloc[:,0].values)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

sel = MRMR(method="RFCQ", regression=False)
sel.fit(X, y)

print(len(sel.relevance_))
sel.variables_= features[:-1]

pd.Series(sel.relevance_, index=sel.variables_).sort_values(
    ascending=True).plot.bar(figsize=(15, 4))
print(sel.relevance_)
plt.title("MID")
plt.show()
