import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def get_category(age):
    cat=''
    if age<=-1: cat='Unknown'
    elif age<=5: cat='Baby'
    elif age<=12: cat='Child'
    elif age<=18: cat='Teenager'
    elif age<=25: cat='Student'
    elif age<=35: cat='Young Adult'
    elif age<=60: cat='Adult'
    else: cat='Elderly'

    return cat


# since scikit ML doesn't allow NULL value, fill value in it    
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df


# delete not necessary things in df
def drop_features(df):
    df.drop(['PassengerID','Name','Ticket'],axis=1,inplace=True)
    return df


# label encoding
def encode_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])

    return df


# preprocess all of above things
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = encode_features(df)
    return df


# 5 fold set with KFold
def exec_kfold(clf,folds=5,X_titanic_df,y_titanic_df):
    kfold = KFold(n_splits=folds)
    scores=[]

    # KFold
    for iter_count,(train_index,test_index) in enumerate(kfold.split(X_titanic_df)):
        X_train,X_test = X_titanic_df.values[train_index],X_titanic_df.values[test_index]
        y_train,y_test = y_titanic_df.values[train_index],y_titanic_df.values[test_index]

        # Classifier train,predict,accuracy
        clf.fit(X_train,y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)
        scores.append(accuracy)
        print('kfold {0} accuracy: {1:.4f}'.format(iter_count,accuracy))

    # mean of 5 fold 
    mean_score = np.mean(scores)
    print('avg accuracy: {0:.4f}'.format(mean_score))




titanic_df = pd.read_csv('titanic/train.csv')

y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)

X_titanic_df = transform_features(X_titanic_df)


# extract test datas from train datas
X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,
test_size=0.2,random_state=11)


# DecisionTree Model
dt_clf = DecisionTreeClassifier(random_state=11)

dt_clf.fit(X_train,y_train)
dt_pred = dt_clf.predict(X_test)
print('Accuracy of DecisionTree Model: {0:.4f}'.format(accuracy_score(y_test,dt_pred))

exec_kfold(dt_clf,folds=5,X_titanic_df,y_titanic_df)


# RandomForest Model
rf_clf = RandomForestClassifier(random_state=11)

rf_clf.fit(X_train,y_train)
rf_pred = rf_clf.predict(X_test)
print('Accuracy of RandomForest Model: {0:.4f}'.format(accuracy_score(y_test,rf_pred))



# LogisticRegression Model
lr_clf = LogisticRegression()

lr_clf.fit(X_train,y_train)
lr_pred = lr_clf.predict(X_train)
print('Accuracy of LogisticRegression Model: {0:.4f}'.format(accuracy_score(y_test,lr_pred))








#plt.figure(figsize=(10,6))

#group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']

#titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
#sns.barplot(x='Age_cat',y='Survived',hue='Sex',data=titanic_df,order=group_names)
#titanic_df.drop('Age_cat',axis=1,inplace=True)
#plt.show()
