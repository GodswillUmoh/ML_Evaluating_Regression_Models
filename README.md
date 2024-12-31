# ML_Evaluating_Regression_Models

## The Big Question is, How do I select the Best Regression Model
_Based on your dataset, this repo explains how to select the best regression model for your Machine Learning_

## The r2_score is used to determine the best regression model
> The R² score, also known as the coefficient of determination, is a statistical metric used to evaluate the performance of a regression model. It quantifies how well the model's predictions align with the actual data.

## Common Names for R² Score:
+ Coefficient of Determination (most formal name).
+ R-Squared (commonly used shorthand).
+ Goodness of Fit Measure (describes its purpose).

## Application of R² Score in the Multiple Linear Regression
[Click Here to View Python Code](https://colab.research.google.com/drive/1Mky8jQRiDqO_MTZL6uxKrpXXkgbZ54aV#scrollTo=xPagAOKDywV4)

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
```python
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

## Application of R² Score in the Decision Tree Regression
[Click Here to View Python Code](https://colab.research.google.com/drive/1fp-4iiJRRHJXQ0hBaGH-tgPx97a6wVvn)

```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
```
```python
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

