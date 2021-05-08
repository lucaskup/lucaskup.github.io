---
layout: post
title: "Model Creation Titanic Synthetic Dataset"
date: 2021-05-01 05:00:00 -0300
img: regression-analysis.svg
description: Jupyter Notebook that creates an ensemble of models for Kaggle Playground Series
tags: [data-science, model-creation]
---

This was my first attempt at a Kaggle Playground Series, I've entered quite late at the competition but it was fun nevertheless.

This notebook contains code for data pre-processing, feature engineering, model optimization (through hyperparameter grid search), and an ensemble of kNN, Random Forest, SVM, LDA, and MLP classifiers. This was far from the best submission in the competition but was under 3% accuracy from the best.

The notebook used in this competition is found [here](https://www.kaggle.com/lucaskup/feature-eng-plus-ensemble-knn-rf-lda-svm-mlp).

I have a [notebook that has sample code for visualization of this dataset]({% post_url 2021-04-21-Titanic-visualization %})

Feel free to use it or suggest improvements on my approach. :) 

## First Notebook for Kaggle Competition

This notebook is was used in two Kaggle competitions, in the classic Titanic competition 
and in the April Tabular Series Competition. It did not achieve the best results, but got a
solid 0.77511 acc (people argue that the best achieveble acc in this competition is ~0.83)
an it scored 0.78449 in the april competition (the first place got 0.81).

TLDR: The datafile is read, we preprocess the data and input nan values using the mean,
onehotencode the categorical features and use an ensemble of LDA, Random Forest, kNN, SVM and MLP classifiers


```python
# Import Libraries
from sklearn.neighbors import VALID_METRICS
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neural_network import MLPClassifier


from sklearn.neighbors import DistanceMetric

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
```


```python
# Read the datasets

datasetTrain = pd.read_csv('data/train.csv')
datasetTest = pd.read_csv('data/test.csv')
datasetTrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: center;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Defines the function that preprocess the data
def preprocessData(df: pd.DataFrame,
                   columnTransformer: ColumnTransformer = None,
                   complementarySetForNull: pd.DataFrame = None) -> (np.ndarray, np.ndarray, ColumnTransformer):
    '''
    Preprocess the data passed in, fill None values, encodes categorical features.

    Parameters:
        df (DataFrame): A pandas dataframe that contains the data to be preprocessed
        columnTransformer (ColumnTransformer): optional column transformer to be used if null it creates a new ct
        complementarySetForNull (Dataframe): An optional df (test) that is used for inputation of missing values.

    Returns:
        X (Numpy Array): The feature array already preprocessed
        Y (Numpy Array): The target value
        ct (Column Transformer): The column transformer used in the preprocess 
    '''
    df.drop(labels=['Name', 'Ticket'],
            axis=1,
            inplace=True)
    if complementarySetForNull is not None:
        # Age fillna with mean age
        completeDF = pd.concat(
            [df, complementarySetForNull]).reset_index(drop=True)
        meanAge = completeDF['Age'].mean()
        meanFare = completeDF['Fare'].mean()
        completeDF = None

        df['Age'] = df['Age'].fillna(meanAge)
        complementarySetForNull['Age'] = complementarySetForNull['Age'].fillna(
            meanAge)

        df['Fare'] = df['Fare'].fillna(meanFare)
        complementarySetForNull['Fare'] = complementarySetForNull['Fare'].fillna(
            meanFare)
    df = df.assign(AgeGroup=df['Age'].apply(
        lambda x: x // 10 if x // 10 <= 6 else 6))
    df = df.assign(hasFamily=(df['SibSp'] + df['Parch']).apply(
        lambda x: 1 if x > 0 else 0))
    df = df.assign(familySize=(df['SibSp'] + df['Parch']).apply(
        lambda x: 0 if pd.isnull(x) else x))
    df = df.assign(hasCabin=df['Cabin'].apply(
        lambda x: 0 if pd.isnull(x) else 1))
    df = df.assign(cabinLetter=df['Cabin'].apply(
        lambda x: '.' if pd.isnull(x) else str(x)[0]))
    df['Fare'] = df['Fare'].apply(
        lambda x: 0 if pd.isnull(x) else x)
    df = df.assign(farePerFamily=df['Fare']/(df['familySize']+1))
    X = df[['Pclass', 'Sex', 'Fare', 'Embarked',
            'AgeGroup', 'hasFamily', 'hasCabin', 'cabinLetter', 'farePerFamily']].values
    if 'Survived' in df.columns:
        Y = np.ravel(df['Survived'])
    else:
        Y = None
    if columnTransformer is None:
        columnTransformer = ColumnTransformer(
            [('encoder', OneHotEncoder(drop='first'), [0, 1, 3, 4, 7]),
             ('minMaxScaler', MinMaxScaler(), [2, 8])], remainder='passthrough')
        X = columnTransformer.fit_transform(X)
    else:
        X = columnTransformer.transform(X)

    return X, Y, columnTransformer
```


```python
# Preprocess the train and the test set and frees the memory of regarding
# the features of the dataset that were not used

X, Y, ct = preprocessData(datasetTrain, complementarySetForNull=datasetTest)

X_test, _, _ = preprocessData(datasetTest, columnTransformer=ct)
passengerIdTest = datasetTest['PassengerId'].values.reshape(-1, 1)

datasetTrain = None
datasetTest = None
ensembleOfModels = []
```

## If you don't have enough hardware...
Sometimes you just don't have enough hardware available, it was my case in April's competition.
My solution was to do a grid search using just a sample of the training set. Usually, you want to use
all the data, but if you are on a budget station (like me) a sample can do the job.
Use the SAMPLE_RATIO (0,1] below to control how much of the training sample you will use.

Feel free to change to 1 if you have enough resources.


```python
# Since we dont have enough hardware to grid search through the entire data
# we will take a i.i.d. sample of 10% of our dataset and will use that
# to grid search and obtain the best hyperparameters for our models

SAMPLE_RATIO = 1
sampleIndexes = np.random.choice(len(Y),
                                 int(len(Y)*SAMPLE_RATIO),
                                 replace=False)
X_sample = X[sampleIndexes]
Y_sample = Y[sampleIndexes]
```

## Grid Search
Some models have hyperparameters, aka values that you have to manually set and that are not subject to optimization during the training phase.
For those, the best practice is to search through some combination of parameters (trial and error) and select the parameters that create the
best model. In sklearn, we have the GridSearchCV that allows us to do a grid search in a k-fold cross-validation setup, we will do that.


```python
# Grid Search Through the Random Forest Classifier

gridParameters = {'n_estimators': [10, 50, 100, 200],
                  'criterion': ['gini', 'entropy']}
gsCV = GridSearchCV(RandomForestClassifier(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best Random Forst Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
```

    Best Random Forst Classifier:
       Score > 0.8069288389513108
       Params > {'criterion': 'entropy', 'n_estimators': 100}



```python

# Grid Search Through the kNN classifier
# to use mahalonobis distance we need to pass the keyword parameters
# V and VI
# in case we want to know the valid distance metrics
# we could run => sorted(VALID_METRICS['brute'])

covParam = np.cov(X.astype(np.float32))
invCovParam = np.linalg.pinv(covParam)

gridParameters = [{'algorithm': ['auto'],
                  'metric': ['minkowski'],
                   'n_neighbors': [5, 10, 20]},
                  {'algorithm': ['brute'],
                   'metric': ['mahalanobis'],
                   'n_neighbors': [5, 10, 20],
                   'metric_params': [{'V': covParam,
                                      'VI': invCovParam}]}]
gsCV = GridSearchCV(KNeighborsClassifier(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best kNN Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
```

    Best kNN Classifier:
       Score > 0.7765917602996255
       Params > {'algorithm': 'auto', 'metric': 'minkowski', 'n_neighbors': 20}



```python
# The LDA models does not have much hyperparameters to tune

model = LinearDiscriminantAnalysis()
cv = cross_validate(model, X_sample, Y_sample, scoring='accuracy', cv=10)

print(np.mean(cv['test_score']))

ensembleOfModels.append(model)
```

    0.7957178526841449



```python
# Lets grid search through the SVM classifier

gridParameters = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto'],  # [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
gsCV = GridSearchCV(svm.SVC(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best SVM:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
```

    Best SVM:
       Score > 0.8203995006242198
       Params > {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}



```python
# Lets grid search through the MLP classifier

gridParameters = {'hidden_layer_sizes': [(5, 5), (10, 5), (10, 10), (15, 10), (15, 15)],
                  'activation': ['logistic', 'relu'],
                  'solver': ['adam'],
                  'alpha': [0.0001, 0.05, 0.005],
                  'learning_rate': ['constant', 'adaptive'],
                  }
gsCV = GridSearchCV(MLPClassifier(max_iter=2500),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best MLP:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
```

    Best MLP:
       Score > 0.8113982521847689
       Params > {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (15, 15), 'learning_rate': 'adaptive', 'solver': 'adam'}


## The prediction...
We now take the best model for each of the five algorithms we used (LDA, kNN, Random Forest, SVM) and we train them with selected
hyperparameters using the whole training set. 

We use each of the five classifiers to predict the target variable. If three or more classifiers vote
on a specific outcome that is the outcome of our ensemble classifier.


```python
# Prepare data for Kaggle Submission
ensembleOfModels = list(map(lambda m: m.fit(X, Y), ensembleOfModels))


predictions = list(map(lambda m: m.predict(X_test), ensembleOfModels))

predictionsEnsemble = predictions[0] + predictions[1] + \
    predictions[2] + predictions[3] + predictions[4]
# predictionsEnsemble = predictionsEnsemble.apply(lambda x: 1 if x >= 3 else 0)
dataForSubmission = pd.DataFrame(np.concatenate((passengerIdTest,
                                                 predictionsEnsemble.reshape(-1, 1)), axis=1), columns=['PassengerId', 'Survived'])
dataForSubmission['Survived'] = dataForSubmission['Survived'].apply(
    lambda x: 1 if x >= 3 else 0)
```


```python
# Creates the submission file
dataForSubmission.to_csv('submission/TabularTitanic.csv',
                         sep=',',
                         decimal='.',
                         index=False)
dataForSubmission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: center;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

