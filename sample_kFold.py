from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits=5)

# StratifiedKFOld
skfold = StratifiedKFold(n_splits=3)

cv_accuracy=[]
print('iris data set size:',features.shape[0])

n_iter=0
for train_index,test_index in kfold.split(features):
    # extract train,test datas from kfold.split()
    X_train,X_test = features[train_index],features[test_index]
    y_train,y_test = label[train_index],label[test_index]

    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter+=1

    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} cross vali accuracy: {1}, train data size: {2}, test data size: {3}'.format(n_iter,accuracy,train_size,test_size))
    print('#{0} test data index:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)

print('mean  of accuracy: ',np.mean(cv_accuracy))
