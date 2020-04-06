import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.base import BaseEstimator
from titanic_prediction import transform_features

class MyDummyClassifier(BaseEstimator):
    
    def fit(self,X,y=None):
        pass

    # predict method simply predicts 1 if female(sex 0), if not 0 if male(sex 1)
    def predict(self,X):
        pred = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            if X['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1

        return pred



titanic_df = pd.read_csv('titanic/train.csv')

y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)
X_titanic_df = transform_features(X_titanic_df)

X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2,random_state=0)


myclf = MyDummyClassifier()
myclf.fit(X_train,y_train)

mypred = myclf.predict(X_test)
print('Dummy Classifier accuracy: {0:.4f}'.format(accuracy_score(y_test,mypred)))



