from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# create DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=156)

# load iris data and classify
iris_data = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=11)

# learn DecisionTree
dt_clf.fit(X_train,y_train)

# extract feature importance
print("Feature importance:\n{0}".format(np.round(dt_clf.feature_importances_,3)))

# mapping importance
for name,value in zip(iris_data.feature_names,dt_clf.feature_importances_):
    print('{0}: {1:.3f}'.format(name,value))

# visualize feature importance for each columns
sns.barplot(x=dt_clf.feature_importances_,y=iris_data.feature_names)
plt.show()

