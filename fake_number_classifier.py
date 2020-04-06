from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self,X,y):
        pass

    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)



digits = load_digits()

y = (digits.target==7).astype(int)
X_train,X_test,y_train,y_test = train_test_split(digits.data,y,random_state=11)


print('size of test set: ',y_test.shape)
print('variance of label 0 and 1: ')
print(pd.Series(y_test).value_counts())

fakeclf=MyFakeClassifier()
fakeclf.fit(X_train,y_train)

fake_pred = fakeclf.predict(X_test)
print('even predicting only 0 : {0:.3f}'.format(accuracy_score(y_test,fake_pred)))
