---
title: "Regression Machine Learning Project"
date: 2020-04-15T15:34:30-04:00
categories:
  - School Project
tags:
  - Machine Learning
  - Python
toc: true
toc_label:
toc_icon: 'bars'
classes: wide

---

# Overview

A manufacturer produces pipes with different specifications.  It wants to use a prediction model to estimate the finished cost of new pipes with pre-determined specifications based on the historic data of pipes which already are manufactured.


<br />

|![image](/assets/images/pipes.png)|
|:--:|
|*There are many attributes that may describe a pipe assembly*|

<br />

We have access to multiple pipe data sets and are asked to predict the finished cost of a given pipe.
The data sets are:

•	Train dataset: including the pipe assembly id and finished price of the pipes used for training the models.

•	Test dataset: including only the pipe assembly id for the pipes which finished price should be predicted.

•	Pipes main spec: including the material of the pipe and technical specifications of the pipes such as diameter, length, etc.

•	Pipes other spec: including the code of other specification of each pipe.

•	Pipes bill of material: including the number and name of the components in each pipe.

•	Compgeneralnumbers: including the number and name of the components types in each pipe.

•	Compweights: including the weight of components types in each pipe.

•	Comptypespec: including the specifications of the components in each pipe.


The following is required:

1.	Data pre-processing (Merging data sets, missing data, outliers,etc).
2.	Creating a validation data set.
3.	Model building:
a.	Linear Regression
b.	Forest Regression
4.	Measuring the Square Logarithmic Error (RMSE) of validation data using each of your prediction models.

<br />

|![image](/assets/images/rmse.png)|
|:--:|
|*Where n is the number of records in the test set, p is your predicted finished cost and a is actual finished cost*|

<br />
  


5.	Select the best model.
6.	Provide prediction of test pipes using the best model.


# Data Pre-processing

Even though it is the first step to creating a machine learning algorithm, it is the most vital step in creating a useful and accurate predictive model. We were handed various csv files each with varying amounts of information. Some of it is redundant, some of it is missing, and it is all scattered across multiple tables. Our goal is to create a single, clean table filled with only the most pertinent data.

In order to do this, though, we will need to import the necessary libraries.

```python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb

import sklearn
from sklearn import decomposition
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
```

Next we will import the data files.

```python
#import csv files
trainV1 = pd.read_csv('train v1.csv') # ask about diff
pipeStructure = pd.read_csv('pipe structure.csv')# data contained in compGenNum
train = pd.read_csv('train.csv') # target values
test = pd.read_csv('test.csv') # pipe id's for calculating the predicted cost
pipeMainSpec = pd.read_csv('pipe main spec.csv') # pipe id and attributes
otherSpec = pd.read_csv('other spec.csv') # majority of data is 0, little impact, ignore
compGenNum = pd.read_csv('compgeneralnumbers.csv') # pipe id and component number (attributes)
compTypeSpec = pd.read_csv('comptypespec.csv')# missing data so may be optional
```
We will first look at attributes held in the *pipeMainSpec* dataframe. A dataframe is an object from the pandas library that is essentially a two-dimensional table.
We should check the table for NaN values as they will interfere with training, since NaN values are not a number and regression calculations can only work with numbers.

```python
#check for NaN values
pipeMainSpec.isna().sum()
```

**OUTPUT:**

    pipe_assembly_id      0
    material_id         279
    diameter              0
    wall                  0
    length                0
    num_bends             0
    bend_radius           0
    end_a_1x              0
    end_a_2x              0
    end_x_1x              0
    end_x_2x              0
    num_boss              0
    num_bracket           0
    other                 0
    dtype: int64


There are many NaN values in the categorical attribute *material_Id*, so remove that column. Then print out the first five rows of the table.

```python
pipeMainSpec = pipeMainSpec.dropna(axis = 1)
pipeMainSpec[:5]
```

**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pipe_assembly_id</th>
      <th>diameter</th>
      <th>wall</th>
      <th>length</th>
      <th>num_bends</th>
      <th>bend_radius</th>
      <th>end_a_1x</th>
      <th>end_a_2x</th>
      <th>end_x_1x</th>
      <th>end_x_2x</th>
      <th>num_boss</th>
      <th>num_bracket</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PA-00001</td>
      <td>12.70</td>
      <td>1.65</td>
      <td>164.0</td>
      <td>5</td>
      <td>38.10</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PA-00002</td>
      <td>6.35</td>
      <td>0.71</td>
      <td>137.0</td>
      <td>8</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PA-00003</td>
      <td>6.35</td>
      <td>0.71</td>
      <td>127.0</td>
      <td>7</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PA-00004</td>
      <td>6.35</td>
      <td>0.71</td>
      <td>137.0</td>
      <td>9</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PA-00005</td>
      <td>19.05</td>
      <td>1.24</td>
      <td>109.0</td>
      <td>4</td>
      <td>50.80</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Next we will look at the *compGenNum* dataframe. It also holds some NaN values, but it is a numerical attribute so we can replace them with zeroes and keep the column. We can also remove the Grand Total column as it is a redundant attribute. 

```python
#replace NaN values with zeroes
compGenNum = compGenNum.fillna(0)
compGenNum = compGenNum.drop('Grand Total', axis = 1)
compGenNum
```



**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pipe_assembly_id</th>
      <th>comp_adaptor</th>
      <th>comp_boss</th>
      <th>comp_elbow</th>
      <th>comp_float</th>
      <th>comp_hfl</th>
      <th>comp_nut</th>
      <th>comp_other</th>
      <th>comp_sleeve</th>
      <th>comp_straight</th>
      <th>comp_tee</th>
      <th>comp_threaded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PA-00001</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PA-00002</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PA-00003</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PA-00004</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PA-00005</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19144</th>
      <td>PA-21193</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19145</th>
      <td>PA-21194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19146</th>
      <td>PA-21195</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19147</th>
      <td>PA-21196</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19148</th>
      <td>PA-21197</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>19149 rows × 12 columns</p>
</div>

We cleaned up two files already so we can combine them into one dataframe called *attributes*.

```python
#merge the two major attribute dataframes
attributes = compGenNum.merge(right = pipeMainSpec, left_index = True,right_index = True, on = "pipe_assembly_id")
attributes
```



**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pipe_assembly_id</th>
      <th>comp_adaptor</th>
      <th>comp_boss</th>
      <th>comp_elbow</th>
      <th>comp_float</th>
      <th>comp_hfl</th>
      <th>comp_nut</th>
      <th>comp_other</th>
      <th>comp_sleeve</th>
      <th>comp_straight</th>
      <th>...</th>
      <th>length</th>
      <th>num_bends</th>
      <th>bend_radius</th>
      <th>end_a_1x</th>
      <th>end_a_2x</th>
      <th>end_x_1x</th>
      <th>end_x_2x</th>
      <th>num_boss</th>
      <th>num_bracket</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PA-00001</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>164.0</td>
      <td>5</td>
      <td>38.10</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PA-00002</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>8</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PA-00003</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>127.0</td>
      <td>7</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PA-00004</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>9</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PA-00005</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>109.0</td>
      <td>4</td>
      <td>50.80</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19144</th>
      <td>PA-21193</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>116.0</td>
      <td>5</td>
      <td>25.40</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19145</th>
      <td>PA-21194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>152.0</td>
      <td>6</td>
      <td>38.10</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19146</th>
      <td>PA-21195</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>52.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19147</th>
      <td>PA-21196</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>3</td>
      <td>50.80</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19148</th>
      <td>PA-21197</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>64.0</td>
      <td>3</td>
      <td>25.40</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>19149 rows × 24 columns</p>
</div>

Here we check where the newly made *attributes* dataframe shares the same pipe assembly ids from the *train* file we were given. A new dataframe called *istrainAttributes* will contain a list of all the pipe assembly ids and whether they are found in both *train* and *attributes*.

```python
istrainAttributes = attributes['pipe_assembly_id'].isin(train.pipe_assembly_id)
istrainAttributes
```


**OUTPUT:**

    0        False
    1         True
    2        False
    3         True
    4        False
    5        False
    6        False
    7        False
    8        False
    9        False
    10       False
    11       False
    12       False
    13       False
    14       False
    15       False
    16       False
    17       False
    18       False
    19       False
    20       False
    21       False
    22        True
    23       False
    24       False
    25       False
    26       False
    27       False
    28       False
    29       False
             ...  
    19119    False
    19120    False
    19121    False
    19122    False
    19123    False
    19124    False
    19125    False
    19126    False
    19127    False
    19128    False
    19129    False
    19130    False
    19131    False
    19132    False
    19133    False
    19134    False
    19135    False
    19136    False
    19137    False
    19138    False
    19139    False
    19140    False
    19141    False
    19142    False
    19143    False
    19144    False
    19145    False
    19146    False
    19147    False
    19148    False
    Name: pipe_assembly_id, Length: 19149, dtype: bool


We will only choose the pipe assembly ids in the *attribute* dataframe that are also found in the *train* dataframe, since the *train* dataframe has the target prices that the machine learning algorithm will actually train on. We will createa dataframe, *trainAttributes*, that will contain the attributes found in both *train* and *pipe assembly id*. Having these target prices for the algorithm to train on makes this a supervised machine learning algorithm as opposed to an unsupervised machine learning algorithm. You can learn more about these two general types of problems [here](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)

```python
trainAttributes = attributes[istrainAttributes == True]
trainAttributes
```


**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pipe_assembly_id</th>
      <th>comp_adaptor</th>
      <th>comp_boss</th>
      <th>comp_elbow</th>
      <th>comp_float</th>
      <th>comp_hfl</th>
      <th>comp_nut</th>
      <th>comp_other</th>
      <th>comp_sleeve</th>
      <th>comp_straight</th>
      <th>...</th>
      <th>length</th>
      <th>num_bends</th>
      <th>bend_radius</th>
      <th>end_a_1x</th>
      <th>end_a_2x</th>
      <th>end_x_1x</th>
      <th>end_x_2x</th>
      <th>num_boss</th>
      <th>num_bracket</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>PA-00002</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>8</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PA-00004</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>9</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PA-00024</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>52.0</td>
      <td>3</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>PA-00052</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>45.0</td>
      <td>2</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>PA-00056</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>92.0</td>
      <td>8</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19010</th>
      <td>PA-21032</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>81.0</td>
      <td>4</td>
      <td>88.90</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19030</th>
      <td>PA-21054</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>157.0</td>
      <td>6</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19080</th>
      <td>PA-21113</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>119.0</td>
      <td>4</td>
      <td>110.00</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19094</th>
      <td>PA-21130</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>96.0</td>
      <td>3</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19095</th>
      <td>PA-21131</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>94.0</td>
      <td>8</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1027 rows × 24 columns</p>
</div>


We will now merge the attributes found in *trainAttributes* and add it to the *train* dataframe. Now the *train* dataframe has a bunch of pipe assemblies with attributes that correspond to a certain price. The machine learning algorithm will use the attributes and corresponding prices to create a model for future pipe assemblies to estimate their prices.

```python
train2 = train.merge(right = trainAttributes, left_index = True, right_index = False, on = "pipe_assembly_id", how = 'outer')
train2
```



**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pipe_assembly_id</th>
      <th>cost</th>
      <th>comp_adaptor</th>
      <th>comp_boss</th>
      <th>comp_elbow</th>
      <th>comp_float</th>
      <th>comp_hfl</th>
      <th>comp_nut</th>
      <th>comp_other</th>
      <th>comp_sleeve</th>
      <th>...</th>
      <th>length</th>
      <th>num_bends</th>
      <th>bend_radius</th>
      <th>end_a_1x</th>
      <th>end_a_2x</th>
      <th>end_x_1x</th>
      <th>end_x_2x</th>
      <th>num_boss</th>
      <th>num_bracket</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>PA-00002</td>
      <td>157.48</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>8.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PA-00004</td>
      <td>157.96</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>9.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PA-00024</td>
      <td>150.50</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>52.0</td>
      <td>3.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>PA-00052</td>
      <td>212.50</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>PA-00056</td>
      <td>205.94</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>92.0</td>
      <td>8.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19010</th>
      <td>PA-21032</td>
      <td>149.66</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>81.0</td>
      <td>4.0</td>
      <td>88.90</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19030</th>
      <td>PA-21054</td>
      <td>137.44</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>157.0</td>
      <td>6.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19080</th>
      <td>PA-21113</td>
      <td>154.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>119.0</td>
      <td>4.0</td>
      <td>110.00</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19094</th>
      <td>PA-21130</td>
      <td>154.08</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>96.0</td>
      <td>3.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19095</th>
      <td>PA-21131</td>
      <td>149.36</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>94.0</td>
      <td>8.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1035 rows × 25 columns</p>
</div>


The pipe assembly ids have no impact on the actual price of the assemblies so we can now remove it for the sake of training.

```python
#drop the unneeded id column
train2 = train2.drop(['pipe_assembly_id'], axis = 1)
```

Certain rows still had NaN values so we will remove them as well.

```python
#drop rows that had all NaN values
train2 = train2.dropna(axis = 0)
train2
```



**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>comp_adaptor</th>
      <th>comp_boss</th>
      <th>comp_elbow</th>
      <th>comp_float</th>
      <th>comp_hfl</th>
      <th>comp_nut</th>
      <th>comp_other</th>
      <th>comp_sleeve</th>
      <th>comp_straight</th>
      <th>...</th>
      <th>length</th>
      <th>num_bends</th>
      <th>bend_radius</th>
      <th>end_a_1x</th>
      <th>end_a_2x</th>
      <th>end_x_1x</th>
      <th>end_x_2x</th>
      <th>num_boss</th>
      <th>num_bracket</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>157.48</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>8.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>157.96</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>9.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>150.50</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>52.0</td>
      <td>3.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>212.50</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>205.94</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>92.0</td>
      <td>8.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>1027 rows × 24 columns</p>
</div>


We can now look for any outliers in the dataset and remove them, as they would skew the training and make the model less accurate.
First we calculate the 25th percentile in the cost column.

```python
cost = train2.cost
p25 = np.percentile(cost,25)
p25
```


**OUTPUT:**

    154.49


Then we calculate the 75th percentile.

```python
p75 = np.percentile(cost,75)
p75
```


**OUTPUT:**

    199.55

Now we will find the lowest cost value that isn't an outlier. So any values lower than this will be considered an outlier.


```python
lower = p25 - 1.5*(p75-p25)
lower
```


**OUTPUT:**

    86.9


And of course we find the highest value too. Any cost higher than this will be considered an outlier as well.

```python
upper = p75 + 1.5*(p75-p25)
upper
```


**OUTPUT:**

    267.14


We remove the outliers from the dataframe using this condition.

```python
train2 = train2[cost<upper]
train2 = train2[cost>lower]
```
    
Since we need to have seperate dataframes for the training data and the label data, we will save and remove the *cost* column. *train2* will be our training data and *cost* will be our label data.

```python
cost = train2.cost
train2 = train2.drop(['cost'], axis = 1)
train2[:5]
```



**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp_adaptor</th>
      <th>comp_boss</th>
      <th>comp_elbow</th>
      <th>comp_float</th>
      <th>comp_hfl</th>
      <th>comp_nut</th>
      <th>comp_other</th>
      <th>comp_sleeve</th>
      <th>comp_straight</th>
      <th>comp_tee</th>
      <th>...</th>
      <th>length</th>
      <th>num_bends</th>
      <th>bend_radius</th>
      <th>end_a_1x</th>
      <th>end_a_2x</th>
      <th>end_x_1x</th>
      <th>end_x_2x</th>
      <th>num_boss</th>
      <th>num_bracket</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>8.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>9.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>52.0</td>
      <td>3.0</td>
      <td>19.05</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>92.0</td>
      <td>8.0</td>
      <td>31.75</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>


Certain attributes have data entries as "Y" and "N". We need them to be numbers so we replace them with zeroes and ones instead.

```python
#replace strings for Y and N as numbers 1 and 0
train2 = train2.replace(to_replace = "N", value = 0)
train2 = train2.replace(to_replace = "Y", value = 1)
train2[:5]
```



**OUTPUT:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp_adaptor</th>
      <th>comp_boss</th>
      <th>comp_elbow</th>
      <th>comp_float</th>
      <th>comp_hfl</th>
      <th>comp_nut</th>
      <th>comp_other</th>
      <th>comp_sleeve</th>
      <th>comp_straight</th>
      <th>comp_tee</th>
      <th>...</th>
      <th>length</th>
      <th>num_bends</th>
      <th>bend_radius</th>
      <th>end_a_1x</th>
      <th>end_a_2x</th>
      <th>end_x_1x</th>
      <th>end_x_2x</th>
      <th>num_boss</th>
      <th>num_bracket</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>8.0</td>
      <td>19.05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>137.0</td>
      <td>9.0</td>
      <td>19.05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>52.0</td>
      <td>3.0</td>
      <td>19.05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>31.75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>92.0</td>
      <td>8.0</td>
      <td>31.75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>


# Create the validation dataset.

Now that we have completed the pre-processing we can split the data into the training and testing datasets. The algorithm needs to test the model against a similarly structured dataset to verify its estimated costs during training. The test dataset can be structured exactly the same if we use the following function to split the training dataset 60/40. The *x_train* contains the attributes it will train on. The *y_train* containst the associated costs. *x_test* and *y_test* have the attributes and costs for validation.

```python
x_train, x_test, y_train, y_test = train_test_split(train2, cost,train_size = 0.6, random_state = 42)
```

# Model building

We can start to create a regression model using the sklearn library.

```python
regr = sklearn.linear_model.LinearRegression()
```

Using the followng function, we can fit the training data to the regression model.

```python
regr.fit(x_train,y_train)
```

We will also fit our data onto a more complex model called a Forest Regression model and see how that compares.

```python
forest = RandomForestRegressor()
forest.fit(x_train,y_train)
```

# Predict costs using the models

Now we can use the two regession models to estimate prices based on the testing data.
First we try the regular regresion model.

```python
y_pred = regr.predict(x_test)
```

Then we try the Forest Regression model.

```python
y_pred2 = forest.predict(x_test)
```

# Error Calculation

We can measure the efficacy of the model by calculating the root mean square error of our models' predictions.

The error calculation of the regular regression model is as follows.

```python
print("Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print('score:', regr.score(x_train,y_train))
```



**OUTPUT:**

    Root mean squared error: 17.50
    Variance score: 0.41
    score: 0.561248522543736
    
    

The error calculation for the Forest Regression model is as follows.

```python
print("Mean squared error: %.2f"
      % metrics.mean_squared_error(y_test, y_pred2))
# Explained variance score: 1 is perfect prediction
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
print('score:', forest.score(x_train,y_train))
```



**OUTPUT:**

    Mean squared error: 273.17
    RMSE:  16.52794712455122
    score: 0.9215509587276817
    

# Conclusion

The RMSE of the Forest Regression model is lower than the linear regression model. And the score, which is actually the R-squared value of the model, is higher for the Forest Regression model. Therefore we can conclude that using the Forest Regression model will be a more accurate way to estimate costs for future pipe assemblies.

We can export our results to csv files.

```python
pred_df = pd.DataFrame(y_pred)
pred_df.to_csv('reg_pred.csv')
pred2_df = pd.DataFrame(y_pred2)
pred2_df.to_csv('forest_pred.csv')
```

In order to visualize our models we can plot the regression and Forest Regression.

```python
plt.scatter(y_test,y_pred)
```


**OUTPUT:**

|![image](/assets/images/output_27_1.png)|
|:--:|
|*Regression Model Scatterplot*|



```python
#plot the forest regression model results
plt.scatter(y_test,y_pred2)
```



**OUTPUT:**

|![image](/assets/images/output_28_1.png)|
|:--:|
|*Forest Regression Model Scatterplot*|

