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




```python
#theres many NaN values in the categorical attritbute material_Id, so remove that column
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




```python
#grand total is a redundant attribute, remove it
compGenNum = compGenNum.drop('Grand Total', axis = 1)
```


```python
#replace NaN values with zeroes
compGenNum = compGenNum.fillna(0)
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




```python
#check where the attributes dataframe shares the same pipe assembly ids from the train dataframe
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




```python
#choose only the pipe assembly ids in the attribute dataframe that are also found in the train dataframe
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




```python
#merge those attributes to the train dataframe
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




```python
#drop the unneeded id column
train2 = train2.drop(['pipe_assembly_id'], axis = 1)
```


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




```python
#calculate the 25th percentile in cost column
cost = train2.cost
p25 = np.percentile(cost,25)
p25
```


**OUTPUT:**

    154.49




```python
#75th percentile
p75 = np.percentile(cost,75)
p75
```


**OUTPUT:**

    199.55




```python
# cost values lower than this are outliers
lower = p25 - 1.5*(p75-p25)
lower
```


**OUTPUT:**

    86.9




```python
# cost above this are outliers
upper = p75 + 1.5*(p75-p25)
upper
```


**OUTPUT:**

    267.14




```python
#remove outliers from train dataset
train2 = train2[cost<upper]
train2 = train2[cost>lower]
cost = train2.cost
```
    


```python
#remove cost column as that is the target label for each row
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




```python
#split the dataset into a dataset to train the model and a dataset to test the model.
x_train, x_test, y_train, y_test = train_test_split(train2, cost,train_size = 0.6, random_state = 42)
```


```python
#create a regression model from sklearn
regr = sklearn.linear_model.LinearRegression()
```


```python
#fit the training attributes and costs to the model
regr.fit(x_train,y_train)
```



**OUTPUT:**
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
#predict costs using the testing dataset
y_pred = regr.predict(x_test)
```


```python
# The Root mean squared error
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
    


```python
#use another more complex regression model
forest = RandomForestRegressor()

#fit the data
forest.fit(x_train,y_train)

#predict costs
y_pred2 = forest.predict(x_test)
```
    


```python
# The mean squared error
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
    


```python
#save cost predictions to csv files
pred_df = pd.DataFrame(y_pred)
pred_df.to_csv('reg_pred.csv')
pred2_df = pd.DataFrame(y_pred2)
pred2_df.to_csv('forest_pred.csv')
```



```python
#plot the regression model results
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
|*Regression Model Scatterplot*|

