import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

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



titanic_df = pd.read_csv('titanic/train.csv')
#print(titanic_df.head(3))

#print(titanic_df.info())

# since scikit ML doesn't allow NULL value, fill value in it
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)

#print('num of NULL: ', titanic_df.isnull().sum().sum())

#print('Variance of SEX:\n', titanic_df['Sex'].value_counts())
#print('Variance of Cabin:\n', titanic_df['Cabin'].value_counts())
#print('Variance of Embarked:\n', titanic_df['Embarked'].value_counts())

#print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())

#sns.barplot(x="Sex",y="Survived",data=titanic_df)
#sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df)
#plt.show()

plt.figure(figsize=(10,6))

group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
sns.barplot(x='Age_cat',y='Survived',hue='Sex',data=titanic_df,order=group_names)
titanic_df.drop('Age_cat',axis=1,inplace=True)
plt.show()
