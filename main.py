import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

# load data sets
x_train_set = pd.read_csv('x_train_set.csv', encoding="latin-1")
y_train_set = pd.read_csv('y_train_set.csv', encoding="latin-1")
x_test_set = pd.read_csv('x_test_set.csv', encoding="latin-1")
y_test_set = pd.read_csv('y_test_set.csv', encoding="latin-1")
y_train_set = y_train_set.values.ravel()
y_test_set = y_test_set.values.ravel()

# parametric identification
rfc = RandomForestClassifier(random_state=3)
rfc.fit(x_train_set,y_train_set)

params = {'n_estimators': sp_randint(50,400),
          'max_features' : sp_randint(2,16),
          'max_depth' : sp_randint(2,10),
          'min_samples_split' : sp_randint(2,25),
          'min_samples_leaf' : sp_randint(1,25),
          'criterion':['gini','entropy']}

rsearch = RandomizedSearchCV(rfc,
                             param_distributions=params,
                             n_iter=50,
                             cv=3, 
                             return_train_score = True,
                             scoring='roc_auc',
                             n_jobs=-1,
                            random_state=5)

rsearch.fit(x_train_set, y_train_set)


print(f'best parameters found:\n{rsearch.best_params_}\n')


print("+"*50)
print("Test Results \n")
y_test_pred  = rfc.predict(x_test_set)
y_test_prob = rfc.predict_proba(x_test_set)[:,1]

print("Confusion Matrix for Test : \n", confusion_matrix(y_test_set, y_test_pred))
print("Accuracy Score for Test : ", accuracy_score(y_test_set, y_test_pred))