import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import numpy as np

x_train = pd.read_csv('x_train_set.csv', encoding="latin-1")
y_train = pd.read_csv('y_train_set.csv', encoding="latin-1")

y_train = y_train.values.ravel()

rf = RandomForestClassifier(random_state=0)
rf.fit(x_train,y_train)

# plot feature importance

# f_i = list(zip(x_train.columns, rf.feature_importances_))
# f_i.sort(key = lambda x : x[1])
# f_i = f_i[-50:]
# plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
#plt.show()

rfe = RFECV(rf,cv=5, verbose=2)

rfe.fit(x_train,y_train)
selected_features = np.array(x_train.columns)[rfe.get_support()]

pd.DataFrame(selected_features).to_csv('selected_features.csv', index=False)
