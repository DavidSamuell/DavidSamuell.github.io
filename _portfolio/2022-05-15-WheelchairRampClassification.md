---
title: Wheelchair Ramps Classification
excerpt: "<img src='/images/portfolio/WheelChairTeaser.jpg' style='height:300px; width:500px'>
<br><br>
On 25th April I took my Data Science Professional Certification from Data Camp, after going through several skill assessments in Python, Statistics, SQL, and Machine Learning along with a coding challenge in data management and exploratory analysis using python I was given this case study as my final project in which I have 24 hours to complete it and present my findings in a video format. I've made some improvements after submitting the project, particularly in the hyperparameter tuning section.
<br><br>
<b>Tags: Classification | Machine Learning | Certification | DataCamp</b>"
date: 2022-05-15
tags:
  - Machine Learning
  - Classification
  - Certification
  - DataCamp
  - Data Science
collection: portfolio
---

On 25th April I took my Data Science Professional Certification from Data Camp, after going through several skill assessments in Python, Statistics, SQL, and Machine Learning along with a coding challenge in data management and exploratory analysis using python I was given this case study as my final project in which I have 24 hours to complete it and present my findings in a video format. I've made some improvements after submitting the project, particularly in the hyperparameter tuning section.

## Problem Statement

Congratulations, you have landed your first job as a data scientist at National Accessibility! National Accessibility currently installs wheelchair ramps for office buildings and schools. However, the marketing manager wants the company to start installing ramps for event venues as well. According to a new survey, approximately 40% of event venues are not wheelchair accessible. However, it is not easy to know whether a venue already has a ramp installed. 

The marketing manager would like to know whether you can develop a model to predict whether an event venue has a wheelchair ramp. To help you with this, he has provided you with a dataset of London venues. This data includes whether the venue has a ramp.

It is a waste of time to contact venues that already have a ramp installed, and it also looks bad for the company. Therefore, it is especially important to exclude locations that already have a ramp. Ideally, at least two-thirds of venues predicted to be without a ramp should not have a ramp. 

The data you will use for this analysis can be accessed here: `"data/event_venues.csv"`

## Load Data


```python
# Use this cell to begin, and add as many cells as you need to complete your analysis!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from statistics import mean

plt.style.use('seaborn')
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
```


```python
df = pd.read_csv("data/event_venues.csv")
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>venue_name</th>
      <th>Loud music / events</th>
      <th>Venue provides alcohol</th>
      <th>Wi-Fi</th>
      <th>supervenue</th>
      <th>U-Shaped_max</th>
      <th>max_standing</th>
      <th>Theatre_max</th>
      <th>Promoted / ticketed events</th>
      <th>Wheelchair accessible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>techspace aldgate east</td>
      <td>False</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>0</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>green rooms hotel</td>
      <td>True</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>40.000000</td>
      <td>120</td>
      <td>80.000000</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>148 leadenhall street</td>
      <td>False</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>0</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>conway hall</td>
      <td>False</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>60</td>
      <td>60.000000</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gridiron building</td>
      <td>False</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>0</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>kimpton fitzroy london</td>
      <td>True</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>6.000000</td>
      <td>0</td>
      <td>112.715867</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>lloyds avenue</td>
      <td>False</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>0</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>public space | members-style bar &amp; dining</td>
      <td>True</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>200</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16 old queen street</td>
      <td>False</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>0</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>siorai bar</td>
      <td>True</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>180</td>
      <td>20.000000</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# No missing value
df.isnull().sum()
```




    venue_name                    0
    Loud music / events           0
    Venue provides alcohol        0
    Wi-Fi                         0
    supervenue                    0
    U-Shaped_max                  0
    max_standing                  0
    Theatre_max                   0
    Promoted / ticketed events    0
    Wheelchair accessible         0
    dtype: int64




```python
df['venue_name'] = df['venue_name'].astype('category')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3910 entries, 0 to 3909
    Data columns (total 10 columns):
     #   Column                      Non-Null Count  Dtype   
    ---  ------                      --------------  -----   
     0   venue_name                  3910 non-null   category
     1   Loud music / events         3910 non-null   bool    
     2   Venue provides alcohol      3910 non-null   int64   
     3   Wi-Fi                       3910 non-null   bool    
     4   supervenue                  3910 non-null   bool    
     5   U-Shaped_max                3910 non-null   float64 
     6   max_standing                3910 non-null   int64   
     7   Theatre_max                 3910 non-null   float64 
     8   Promoted / ticketed events  3910 non-null   bool    
     9   Wheelchair accessible       3910 non-null   bool    
    dtypes: bool(5), category(1), float64(2), int64(2)
    memory usage: 192.6 KB
    


```python
len(df)
```




    3910



First before we perform anything let's split our data for training and testing, we will leave the testing data alone until the very end once we've determined a model we think is best. This prevent overfitting to the test data and provides a non bias overview of our model generalization capability.


```python
df_train, df_test = train_test_split(df, test_size=0.2, random_state = 0)
```


```python
target = 'Wheelchair accessible'
```

## EDA


```python
# First lets try to spot any potential outliers in our numerical data using histogram and summary statistics.
df_train.hist(figsize=(8,8), xrot=45, bins=20)
plt.show()
```


    
![png](\images\portfolio\Wheelchair Ramp\output_11_0.png)
    



```python
df_train.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Venue provides alcohol</th>
      <th>U-Shaped_max</th>
      <th>max_standing</th>
      <th>Theatre_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.715473</td>
      <td>34.469905</td>
      <td>111.485934</td>
      <td>111.876338</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.451261</td>
      <td>20.041665</td>
      <td>249.709427</td>
      <td>119.055118</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>35.045455</td>
      <td>0.000000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>35.045455</td>
      <td>50.000000</td>
      <td>112.715867</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>35.045455</td>
      <td>120.000000</td>
      <td>112.715867</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>900.000000</td>
      <td>7500.000000</td>
      <td>2500.000000</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the summary statistics we can see that the distributions in `u-Shaped_max`, `max_standing`, and `Theatre_max` are quite large. This could be some consideration whether or not we should perform feature scaling. Furthermore the maximum value of the feature are really far out from the mean which could suggest outliers. However, to be able to say that it is an outlier require us to have a good understanding at the source of our data. For this case I would say that these are not as it make sense that there are some amount of venue which have a total capacity far larger from the rest.

### Imbalance Class?


```python
# Let's check if there is any imbalances in our target variable
sns.countplot(x = target, data = df_train)
```




    <AxesSubplot:xlabel='Wheelchair accessible', ylabel='count'>




    
![png](\images\portfolio\Wheelchair Ramp\output_15_1.png)
    

We can see that the number of venue accessible to wheelchair and those that are not are pretty balance so we don't have to perform any resampling.

### Segment and group by the target feature


```python
df_train[target] = df_train[target].astype('category')
```

    <ipython-input-11-badd2281d496>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_train[target] = df_train[target].astype('category')
    


```python
num_cols = ['U-Shaped_max', 'max_standing', 'Theatre_max']
for col in num_cols:
    ax = sns.boxplot(y = target, x = col, data=df_train)
    ax.set_xlim(0,500)
    plt.show()
```


    
![png](\images\portfolio\Wheelchair Ramp\output_19_0.png)
    



    
![png](\images\portfolio\Wheelchair Ramp\output_19_1.png)
    



    
![png](\images\portfolio\Wheelchair Ramp\output_19_2.png)
    


From the box plot we can see that there are a lot of so called `outlier` in our numerical features and also the distribution for `Theatre_max` and `U-shaped_max` are really dispersed. But more importantly in a glance we can see that a larger proportion of venue who have larger max and standing capacity tends to be Wheelchair accessible. Keeping these in mind these 2 features might be important in us predicting venue with wheelchair accessibility.

### Segment Categorical features by the target classes


```python
categorical = ['Venue provides alcohol', 'Loud music / events', 'Wi-Fi', 'supervenue', 'Promoted / ticketed events']
```


```python
for col in categorical:
        g = sns.catplot(x = col, kind='count', col = target, data=df_train, sharey=False)
```


    
![png](\images\portfolio\Wheelchair Ramp\output_23_0.png)
    



    
![png](\images\portfolio\Wheelchair Ramp\output_23_1.png)
    



    
![png](\images\portfolio\Wheelchair Ramp\output_23_2.png)
    



    
![png](\images\portfolio\Wheelchair Ramp\output_23_3.png)
    



    
![png](\images\portfolio\Wheelchair Ramp\output_23_4.png)
    


We can see that there are some difference in how the target variable is distributed in venue that provides `alcohol` and venue that hosts `promoted/ticketed events`. We see that in venue that hosts `promoted/ticketed` events tend to be wheelchair accesible, while venue that doesn't provide alcohol tend to be non accessible to wheelchair. This shows that these features might be a good indicator of the target.

### Correlation Matrix


```python
df_train[target] = df_train[target].astype('bool')
corr = df_train.corr()
plt.figure(figsize=(6,6))
sns.heatmap(corr, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
plt.show()
```

    <ipython-input-15-e770310b8beb>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_train[target] = df_train[target].astype('bool')
    


    
![png](\images\portfolio\Wheelchair Ramp\output_26_1.png)
    


From the correlation matrix we can see that there isn't any strong positive or negative linear correlation between the features and our target variable `Wheelchair accesible`.

## Feature Pre-Processing


```python
df_train.dtypes
```




    venue_name                    category
    Loud music / events               bool
    Venue provides alcohol           int64
    Wi-Fi                             bool
    supervenue                        bool
    U-Shaped_max                   float64
    max_standing                     int64
    Theatre_max                    float64
    Promoted / ticketed events        bool
    Wheelchair accessible             bool
    dtype: object




```python
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
```


```python
X = df_train.iloc[:, 1:-1]
y = df_train[target].replace({False: 1, True: 0})

```

You might be wondering why I set the False target variable which indicate venue that is not accesible to wheelchair to `1` instead of `0`. Well the reason is simple if we remember earlier one of the main requirements given by the Sales Manager for this project is to minimize predicting a venue as non-accessible to wheelchair ramp when it is in fact accessible, this means that we want to optimize the precision for when the target variable is False. To do so we need to set our False target variable as the positive variable (denoted as `1`) since Sk-learn can only calculate the precision of the positive class. 

### Scaling

As we've seen from our previous data exploratory analysis we see that the features in our data have varying distributions and even indication of so called "outliers", this might effect the performance of distance based algorithm (KNN, SVM) and gradient descent algorithm (logistic regression). To fix this it is important to scale our data before feeding them to our model, I will try 2 methods of scaling standard scaling: Standard Scaling and MinMaxScaler. Standard scaling will help reduce the importance of outliers by scaling the distribution to an std = 1. MinMaxScaler on the other hand doesn't reduce the important of outliers and are less distruptive to the information of the original data it will scale the data to a default range of 0 - 1.

We will simply try both of these scaling method and see which yields best result on our model.


```python
# Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
```


```python
X_scale = pd.DataFrame(X_scale, index = X.index, columns = X.columns)
X_scale.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loud music / events</th>
      <th>Venue provides alcohol</th>
      <th>Wi-Fi</th>
      <th>supervenue</th>
      <th>U-Shaped_max</th>
      <th>max_standing</th>
      <th>Theatre_max</th>
      <th>Promoted / ticketed events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>788</th>
      <td>-0.752041</td>
      <td>-1.585751</td>
      <td>0.270315</td>
      <td>-0.263442</td>
      <td>0.028722</td>
      <td>-0.446534</td>
      <td>0.007053</td>
      <td>-0.788928</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>-0.752041</td>
      <td>-1.585751</td>
      <td>0.270315</td>
      <td>-0.263442</td>
      <td>-0.223066</td>
      <td>-0.326375</td>
      <td>-0.687828</td>
      <td>-0.788928</td>
    </tr>
    <tr>
      <th>519</th>
      <td>-0.752041</td>
      <td>-1.585751</td>
      <td>0.270315</td>
      <td>3.795901</td>
      <td>0.028722</td>
      <td>-0.406481</td>
      <td>0.007053</td>
      <td>-0.788928</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_scale.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loud music / events</th>
      <th>Venue provides alcohol</th>
      <th>Wi-Fi</th>
      <th>supervenue</th>
      <th>U-Shaped_max</th>
      <th>max_standing</th>
      <th>Theatre_max</th>
      <th>Promoted / ticketed events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.128000e+03</td>
      <td>3.128000e+03</td>
      <td>3.128000e+03</td>
      <td>3.128000e+03</td>
      <td>3.128000e+03</td>
      <td>3.128000e+03</td>
      <td>3.128000e+03</td>
      <td>3.128000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-2.661980e-17</td>
      <td>-1.171271e-18</td>
      <td>-3.510264e-17</td>
      <td>-2.413528e-16</td>
      <td>-1.001900e-15</td>
      <td>-2.259488e-16</td>
      <td>-1.323450e-15</td>
      <td>1.426821e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000160e+00</td>
      <td>1.000160e+00</td>
      <td>1.000160e+00</td>
      <td>1.000160e+00</td>
      <td>1.000160e+00</td>
      <td>1.000160e+00</td>
      <td>1.000160e+00</td>
      <td>1.000160e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-7.520409e-01</td>
      <td>-1.585751e+00</td>
      <td>-3.699385e+00</td>
      <td>-2.634420e-01</td>
      <td>-1.620379e+00</td>
      <td>-4.465340e-01</td>
      <td>-9.230507e-01</td>
      <td>-7.889275e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.520409e-01</td>
      <td>-1.585751e+00</td>
      <td>2.703152e-01</td>
      <td>-2.634420e-01</td>
      <td>2.872223e-02</td>
      <td>-4.465340e-01</td>
      <td>-2.677872e-01</td>
      <td>-7.889275e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-7.520409e-01</td>
      <td>6.306160e-01</td>
      <td>2.703152e-01</td>
      <td>-2.634420e-01</td>
      <td>2.872223e-02</td>
      <td>-2.462693e-01</td>
      <td>7.052730e-03</td>
      <td>-7.889275e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.329715e+00</td>
      <td>6.306160e-01</td>
      <td>2.703152e-01</td>
      <td>-2.634420e-01</td>
      <td>2.872223e-02</td>
      <td>3.410135e-02</td>
      <td>7.052730e-03</td>
      <td>1.267544e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.329715e+00</td>
      <td>6.306160e-01</td>
      <td>2.703152e-01</td>
      <td>3.795901e+00</td>
      <td>4.319344e+01</td>
      <td>2.959318e+01</td>
      <td>2.006218e+01</td>
      <td>1.267544e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_norm = pd.DataFrame(X_norm, index = X.index, columns = X.columns)
X_norm.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loud music / events</th>
      <th>Venue provides alcohol</th>
      <th>Wi-Fi</th>
      <th>supervenue</th>
      <th>U-Shaped_max</th>
      <th>max_standing</th>
      <th>Theatre_max</th>
      <th>Promoted / ticketed events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>788</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.036799</td>
      <td>0.000000</td>
      <td>0.044322</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.031180</td>
      <td>0.004000</td>
      <td>0.011209</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>519</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.036799</td>
      <td>0.001333</td>
      <td>0.044322</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_norm.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loud music / events</th>
      <th>Venue provides alcohol</th>
      <th>Wi-Fi</th>
      <th>supervenue</th>
      <th>U-Shaped_max</th>
      <th>max_standing</th>
      <th>Theatre_max</th>
      <th>Promoted / ticketed events</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
      <td>3128.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.361253</td>
      <td>0.715473</td>
      <td>0.931905</td>
      <td>0.064898</td>
      <td>0.036158</td>
      <td>0.014865</td>
      <td>0.043986</td>
      <td>0.383632</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.480441</td>
      <td>0.451261</td>
      <td>0.251948</td>
      <td>0.246385</td>
      <td>0.022318</td>
      <td>0.033295</td>
      <td>0.047660</td>
      <td>0.486348</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.036799</td>
      <td>0.000000</td>
      <td>0.031225</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.036799</td>
      <td>0.006667</td>
      <td>0.044322</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.036799</td>
      <td>0.016000</td>
      <td>0.044322</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Model Training

### K-Fold Cross Validation 

To evaluate our model performance on our training set we will use K-Fold cross validation more specifically 5-Fold since our data are quite limited. K-Fold validation will allow us to get a fairly accurate overview of our model performance and generalization capability since it trains our model using different part of our data and it works well with small size dataset due to the repeated cross fold validation.


```python
cv = KFold(n_splits = 5, random_state = 0, shuffle=True)
```


```python
def get_score(model, X, y, metric):
    return cross_val_score(model, X = X, y = y, scoring = metric, cv = cv, n_jobs = -1)
```

For the next part I'll implement a bunch of different algorithm with their default parameter and see which one works best.

### Logistic Regresssion


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X, y)

logreg_score = mean(get_score(lr, X, y, 'precision'))
print(logreg_score)
```

    c:\Users\David\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    0.6084089246120791
    


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X_scale, y)

scale_logreg_score = mean(get_score(lr, X_scale, y, 'precision'))
print(scale_logreg_score)
```

    0.6063055862836805
    


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X_norm, y)

norm_logreg_score = mean(get_score(lr, X_norm, y, 'precision'))
print(norm_logreg_score)
```

    0.6056049667216648
    

### Decision Tree


```python
dt = DecisionTreeClassifier(random_state = 0)
dt.fit(X, y)

dt_score = mean(get_score(dt, X, y, 'precision'))
print(dt_score)
```

    0.6321141459006742
    


```python
dt.get_n_leaves(), len(X)
```




    (825, 3128)




```python
# let's add some contrain to our tree to prevent overfitting
dt = DecisionTreeClassifier(random_state = 0, min_samples_leaf = 25)
dt.fit(X, y)

dt_score = mean(get_score(dt, X, y, 'precision'))
print(dt_score)
```

    0.6282740022143144
    


```python
dt.get_n_leaves(), len(X)
```




    (83, 3128)



#### Feature Importance
Feature importance will give a good idea of which features are most useful to the tree when splitting the node to get a better perfomance.


```python
def feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
```


```python
fi = feat_importance(dt, X)
fi[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Theatre_max</td>
      <td>0.528387</td>
    </tr>
    <tr>
      <th>5</th>
      <td>max_standing</td>
      <td>0.241195</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Promoted / ticketed events</td>
      <td>0.070997</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Venue provides alcohol</td>
      <td>0.061862</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Loud music / events</td>
      <td>0.045346</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U-Shaped_max</td>
      <td>0.041549</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wi-Fi</td>
      <td>0.005378</td>
    </tr>
    <tr>
      <th>3</th>
      <td>supervenue</td>
      <td>0.005286</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi)
```




    <AxesSubplot:ylabel='cols'>




    
![png](\images\portfolio\Wheelchair Ramp\output_58_1.png)
    



```python
# Let's try removing some of these features that have low importance score as they might no be that relevant in our prediction.
to_keep = fi[fi.imp>0.01].cols
X_imp_dt = X[to_keep]
```


```python
dt_imp = DecisionTreeClassifier(random_state = 0, min_samples_leaf = 25)
dt_imp.fit(X_imp_dt, y)

dt_score = mean(get_score(dt_imp, X_imp_dt, y, 'precision'))
print(dt_score)
```

    0.6281904981143176
    


```python
fi = feat_importance(dt_imp, X_imp_dt)
fi[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Theatre_max</td>
      <td>0.532250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max_standing</td>
      <td>0.243995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Promoted / ticketed events</td>
      <td>0.072290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Venue provides alcohol</td>
      <td>0.062989</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Loud music / events</td>
      <td>0.046171</td>
    </tr>
    <tr>
      <th>5</th>
      <td>U-Shaped_max</td>
      <td>0.042305</td>
    </tr>
  </tbody>
</table>
</div>



### Random Forest Classifier


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf = RandomForestClassifier(n_estimators = 500, min_samples_leaf = 0.1, random_state = 0)
rf.fit(X, y)

rf_score = mean(get_score(rf, X, y, 'precision'))
print(rf_score)
```

    0.6546381919484513
    


```python
rf_fi = feat_importance(rf, X)
rf_fi[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>max_standing</td>
      <td>0.289665</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Theatre_max</td>
      <td>0.217849</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Promoted / ticketed events</td>
      <td>0.212952</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Venue provides alcohol</td>
      <td>0.174762</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U-Shaped_max</td>
      <td>0.067801</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Loud music / events</td>
      <td>0.036971</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wi-Fi</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>supervenue</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Let's remove `Wi-Fi` and `supervenue` since they have no contribution to the random forest.


```python
rf_to_keep = rf_fi[rf_fi.imp>0.05].cols
X_imp_rf = X[rf_to_keep]
```


```python
rf_imp = RandomForestClassifier(n_estimators = 500, min_samples_leaf = 0.1, random_state = 0)
rf_imp.fit(X_imp_rf, y)

rf_score = mean(get_score(rf_imp, X_imp_rf, y, 'precision'))
print(rf_score)
```

    0.6496156104808377
    


```python
rf_fi = feat_importance(rf_imp, X_imp_rf)
rf_fi[:10]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>max_standing</td>
      <td>0.342606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Theatre_max</td>
      <td>0.287633</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Promoted / ticketed events</td>
      <td>0.203867</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Venue provides alcohol</td>
      <td>0.136621</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U-Shaped_max</td>
      <td>0.029273</td>
    </tr>
  </tbody>
</table>
</div>



### Boosting


```python
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(base_estimator = dt, n_estimators = 400, random_state = 0)

ab_score = mean(get_score(adb, X, y, 'precision'))
print(ab_score)
```

    0.6279814689432012
    


```python
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators = 400, random_state = 0)

gb_score = mean(get_score(gbc, X, y, 'precision'))
print(gb_score)
```

    0.638820254532681
    


```python
# Schochastic Gradient Boosting
sgb = GradientBoostingClassifier(n_estimators = 400, subsample = 0.8, max_features = 0.2, random_state = 0)
sgb.fit(X, y)

sgb_score = mean(get_score(sgb, X, y, 'precision'))
print(sgb_score)
```

    0.6410017296544348
    


```python
fi = feat_importance(sgb, X)
fi[:10]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Theatre_max</td>
      <td>0.329313</td>
    </tr>
    <tr>
      <th>5</th>
      <td>max_standing</td>
      <td>0.257683</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U-Shaped_max</td>
      <td>0.164669</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Promoted / ticketed events</td>
      <td>0.083803</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Venue provides alcohol</td>
      <td>0.065942</td>
    </tr>
    <tr>
      <th>3</th>
      <td>supervenue</td>
      <td>0.036859</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wi-Fi</td>
      <td>0.033934</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Loud music / events</td>
      <td>0.027796</td>
    </tr>
  </tbody>
</table>
</div>



It seems that all of the features are relevant when using gradient boosting so we won't remove any feature like we did in our decision tree and random forest classifier as it will only lower the performance

### KNN


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

knn_score = mean(get_score(knn, X, y, 'precision'))
print(knn_score)
```

    0.6198340928379169
    


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

scale_knn_score = mean(get_score(knn, X_scale, y, 'precision'))
print(scale_knn_score)
```

    0.6327969183534807
    


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

norm_knn_score = mean(get_score(knn, X_norm, y, 'precision'))
print(norm_knn_score)
```

    0.6317485643334988
    

### SVM


```python
from sklearn.svm import SVC, LinearSVC
svc = SVC(random_state = 0)

svc_score = mean(get_score(svc, X, y, 'precision'))
print(svc_score)
```

    0.6269802103034781
    


```python
from sklearn.svm import SVC, LinearSVC
svc = SVC(random_state = 0)

scale_svc_score = mean(get_score(svc, X_scale, y, 'precision'))
print(scale_svc_score)
```

    0.6180157484965274
    


```python
from sklearn.svm import SVC, LinearSVC
svc = SVC(random_state = 0)

norm_svc_score = mean(get_score(svc, X_norm, y, 'precision'))
print(norm_svc_score)
```

    0.6227870276502987
    


```python
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
              'Ada Boost', 'Gradient Boosting', 'Stochastic Gradient Boosting', 
              'Scaled KNN', 'SVM'],
    'Precision Score': [logreg_score, dt_score , rf_score, 
              ab_score, gb_score, sgb_score, 
              scale_knn_score, svc_score]})
models.sort_values(by='Precision Score', ascending=False)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Precision Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>0.649616</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Stochastic Gradient Boosting</td>
      <td>0.641002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gradient Boosting</td>
      <td>0.638820</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Scaled KNN</td>
      <td>0.632797</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>0.628190</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ada Boost</td>
      <td>0.627981</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SVM</td>
      <td>0.626980</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.608409</td>
    </tr>
  </tbody>
</table>
</div>



Based on the comparisons we can find that tree based models like Random Forest and Gradient Boosing yields the highest precision score. Now let's take two of our best model and try to optimize them with some hyperparameter tuning.

## Hyperparameter Tuning.

Since this is a fairly small dataset we won't be using any advanced informed search algorithm like bayesian optimizaiton or genetic algorithm. We'll simply be using the trusty old GridSearch and RandomSearch.  


```python
# Import the necessary module.
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
```

### Random Forest

Random forest are extremely resilient to hyperparameter choices and should not overfit even with large number of tree as they are independent from one another. To tune the model I first perform Randomized Search CV to get an estimation of the optimal hyperparameters, then I narrow down the range of values for each hyperparameters and perform a Grid Search CV.


```python
rs_param_grid = {
    "n_estimators": list((range(300, 500))),
    "max_depth": list((range(4, 20, 2))),
    "min_samples_leaf": list((range(4, 16, 2))),
    "min_samples_split": list((range(10, 50, 5))),
    "max_features": ['auto', 'sqrt']
}

rf = RandomForestClassifier(random_state = 0)

rf_rs = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rs_param_grid,
    cv=cv,  # Number of folds
    n_iter=100,  # Number of parameter candidate settings to sample
    verbose=0,  # The higher this is, the more messages are outputed
    scoring="precision",  # Metric to evaluate performance
    random_state=0,
    n_jobs= -1
)

rf_rs.fit(X, y)
```




    RandomizedSearchCV(cv=KFold(n_splits=5, random_state=0, shuffle=True),
                       estimator=RandomForestClassifier(random_state=0), n_iter=100,
                       n_jobs=-1,
                       param_distributions={'max_depth': [4, 6, 8, 10, 12, 14, 16,
                                                          18],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [4, 6, 8, 10, 12,
                                                                 14],
                                            'min_samples_split': [10, 15, 20, 25,
                                                                  30, 35, 40, 45],
                                            'n_estimators': [300, 301, 302, 303,
                                                             304, 305, 306, 307,
                                                             308, 309, 310, 311,
                                                             312, 313, 314, 315,
                                                             316, 317, 318, 319,
                                                             320, 321, 322, 323,
                                                             324, 325, 326, 327,
                                                             328, 329, ...]},
                       random_state=0, scoring='precision')




```python
rf_rs.best_params_, rf_rs.best_score_
```




    ({'n_estimators': 303,
      'min_samples_split': 30,
      'min_samples_leaf': 4,
      'max_features': 'sqrt',
      'max_depth': 16},
     0.6592529494144459)




```python
rs_param_grid = {
    "n_estimators": list((range(200, 450, 50))),
    "max_depth": list((range(10, 22, 2))),
    "min_samples_leaf": list((range(2, 14, 2))),
    "min_samples_split": list((range(10, 50, 5))),
    "max_features": ['sqrt']
}

rf = RandomForestClassifier(random_state = 0)

rf_rs = GridSearchCV(
    estimator=rf,
    param_grid=rs_param_grid,
    cv=cv,  # Number of folds 
    verbose=0,  # The higher this is, the more messages are outputed
    scoring="precision",  # Metric to evaluate performance
    n_jobs= -1
)

rf_rs.fit(X, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=0, shuffle=True),
                 estimator=RandomForestClassifier(random_state=0), n_jobs=-1,
                 param_grid={'max_depth': [10, 12, 14, 16, 18, 20],
                             'max_features': ['sqrt'],
                             'min_samples_leaf': [2, 4, 6, 8, 10, 12],
                             'min_samples_split': [10, 15, 20, 25, 30, 35, 40, 45],
                             'n_estimators': [200, 250, 300, 350, 400]},
                 scoring='precision')




```python
rf_rs.best_params_, rf_rs.best_score_
```




    ({'max_depth': 16,
      'max_features': 'sqrt',
      'min_samples_leaf': 2,
      'min_samples_split': 25,
      'n_estimators': 300},
     0.6656854437117669)




```python
rf_tuned = RandomForestClassifier(n_estimators= 300, min_samples_split= 25, min_samples_leaf = 2, max_features= 'sqrt', max_depth= 16, random_state=0)
rf_tuned.fit(X, y)

rf_prec_score = mean(get_score(rf_tuned, X, y, 'precision'))
rf_acc_score = mean(get_score(rf_tuned, X, y, 'accuracy'))
print("Precision: {}, Accuracy: {}".format(rf_prec_score, rf_acc_score))
```

    Precision: 0.6656854437117667, Accuracy: 0.6704030670926517
    


```python
rf_fi = feat_importance(rf_tuned, X)
rf_fi[:10]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Theatre_max</td>
      <td>0.337579</td>
    </tr>
    <tr>
      <th>5</th>
      <td>max_standing</td>
      <td>0.302497</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U-Shaped_max</td>
      <td>0.137787</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Promoted / ticketed events</td>
      <td>0.072131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Venue provides alcohol</td>
      <td>0.061466</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Loud music / events</td>
      <td>0.030885</td>
    </tr>
    <tr>
      <th>3</th>
      <td>supervenue</td>
      <td>0.029704</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wi-Fi</td>
      <td>0.027951</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(rf_fi)
```




    <AxesSubplot:ylabel='cols'>




    
![png](\images\portfolio\Wheelchair Ramp\output_97_1.png)
    


###  Stochastic Gradient Boosting

Performing optimization to SGB model are trickier compared to random forest model, they are extremely sensitive to the choice of hyperparameters and there's nothing stopping us from overfitting as we increase the number of tree. The following steps are based on this useful [article](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/) which give a comprehensive guide to tuning a Gradient Boosting model.


```python
param_test1 = {'n_estimators':range(10,110,10)}
sgb = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=30, min_samples_leaf=4, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
gsearch1 = GridSearchCV(estimator = sgb , param_grid = param_test1, scoring='precision', n_jobs=-1, cv=cv)
gsearch1.fit(X, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=0, shuffle=True),
                 estimator=GradientBoostingClassifier(max_depth=8,
                                                      max_features='sqrt',
                                                      min_samples_leaf=4,
                                                      min_samples_split=30,
                                                      random_state=0,
                                                      subsample=0.8),
                 n_jobs=-1, param_grid={'n_estimators': range(20, 110, 10)},
                 scoring='precision')




```python
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
```




    ({'mean_fit_time': array([0.12240252, 0.18079839, 0.22800064, 0.29360037, 0.31699891,
             0.42240057, 0.47680054, 0.53859844, 0.50699825]),
      'std_fit_time': array([0.00831587, 0.00875006, 0.00384716, 0.02402232, 0.02019837,
             0.02303587, 0.01146052, 0.02033394, 0.03014674]),
      'mean_score_time': array([0.02039647, 0.00740061, 0.00860014, 0.0087996 , 0.0084013 ,
             0.00959821, 0.00859933, 0.00939989, 0.0066009 ]),
      'std_score_time': array([0.01032626, 0.00135684, 0.00206077, 0.00222705, 0.00102076,
             0.00320091, 0.0008004 , 0.00101937, 0.00101976]),
      'param_n_estimators': masked_array(data=[20, 30, 40, 50, 60, 70, 80, 90, 100],
                   mask=[False, False, False, False, False, False, False, False,
                         False],
             fill_value='?',
                  dtype=object),
      'params': [{'n_estimators': 20},
       {'n_estimators': 30},
       {'n_estimators': 40},
       {'n_estimators': 50},
       {'n_estimators': 60},
       {'n_estimators': 70},
       {'n_estimators': 80},
       {'n_estimators': 90},
       {'n_estimators': 100}],
      'split0_test_score': array([0.67630058, 0.66959064, 0.66951567, 0.66666667, 0.67422096,
             0.68091168, 0.67705382, 0.67323944, 0.66946779]),
      'split1_test_score': array([0.63380282, 0.63043478, 0.62933333, 0.63487738, 0.6398892 ,
             0.63858696, 0.6344086 , 0.6284153 , 0.62942779]),
      'split2_test_score': array([0.64179104, 0.64371257, 0.63988095, 0.64477612, 0.64011799,
             0.64094955, 0.64450867, 0.64222874, 0.64431487]),
      'split3_test_score': array([0.69496855, 0.70253165, 0.69811321, 0.69811321, 0.69349845,
             0.6863354 , 0.68965517, 0.68847352, 0.6875    ]),
      'split4_test_score': array([0.60422961, 0.61261261, 0.6105919 , 0.6125    , 0.60625   ,
             0.609375  , 0.61370717, 0.61419753, 0.60869565]),
      'mean_test_score': array([0.65021852, 0.65177645, 0.64948701, 0.65138668, 0.65079532,
             0.65123172, 0.65186669, 0.64931091, 0.64788122]),
      'std_test_score': array([0.03205719, 0.03145706, 0.03090932, 0.02913855, 0.03029704,
             0.02874288, 0.02784757, 0.02766231, 0.02801565]),
      'rank_test_score': array([6, 2, 7, 3, 5, 4, 1, 8, 9])},
     {'n_estimators': 80},
     0.6518666869112405)




```python
param_test2 = {'max_depth':range(4,14,2), 'min_samples_split':range(5, 35, 5)}
sgb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=0)
gsearch2 = GridSearchCV(estimator = sgb, param_grid = param_test2, scoring='precision', n_jobs=-1, cv=cv)
gsearch2.fit(X, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=0, shuffle=True),
                 estimator=GradientBoostingClassifier(max_features='sqrt',
                                                      n_estimators=80,
                                                      random_state=0,
                                                      subsample=0.8),
                 n_jobs=-1,
                 param_grid={'max_depth': range(4, 14, 2),
                             'min_samples_split': range(5, 35, 5)},
                 scoring='precision')




```python
gsearch2.best_params_, gsearch2.best_score_
```




    ({'max_depth': 10, 'min_samples_split': 20}, 0.6539445149040549)




```python
param_test3 = {'max_depth':range(6,16,2), 'min_samples_split':range(5, 35, 5), 'min_samples_leaf':range(2, 20, 2)}
sgb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=0)
gsearch3 = GridSearchCV(estimator = sgb, param_grid = param_test3, scoring='precision', n_jobs=-1, cv=cv)
gsearch3.fit(X, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=0, shuffle=True),
                 estimator=GradientBoostingClassifier(max_features='sqrt',
                                                      n_estimators=80,
                                                      random_state=0,
                                                      subsample=0.8),
                 n_jobs=-1,
                 param_grid={'max_depth': range(6, 16, 2),
                             'min_samples_leaf': range(2, 20, 2),
                             'min_samples_split': range(5, 35, 5)},
                 scoring='precision')




```python
gsearch3.best_params_, gsearch3.best_score_
```




    ({'max_depth': 14, 'min_samples_leaf': 18, 'min_samples_split': 5},
     0.6587001254347038)




```python
param_test4 = {"max_features": range(1, 9, 1)}
sgb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=14, min_samples_leaf=18, min_samples_split=5, subsample=0.8, random_state=0)
gsearch4 = GridSearchCV(estimator = sgb, param_grid = param_test4, scoring='precision', n_jobs=-1, cv=cv)
gsearch4.fit(X, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=0, shuffle=True),
                 estimator=GradientBoostingClassifier(max_depth=14,
                                                      min_samples_leaf=18,
                                                      min_samples_split=5,
                                                      n_estimators=80,
                                                      random_state=0,
                                                      subsample=0.8),
                 n_jobs=-1, param_grid={'max_features': range(1, 9)},
                 scoring='precision')




```python
gsearch4.best_params_, gsearch4.best_score_
```




    ({'max_features': 2}, 0.6587001254347038)




```python
param_test5 = {"subsample": np.arange(0.6, 0.9, 0.05)}
sgb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=14, min_samples_leaf=18, min_samples_split=5, max_features = 2, random_state=0)
gsearch5 = GridSearchCV(estimator = sgb, param_grid = param_test5, scoring='precision', n_jobs=-1, cv=cv)
gsearch5.fit(X, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=0, shuffle=True),
                 estimator=GradientBoostingClassifier(max_depth=14, max_features=2,
                                                      min_samples_leaf=18,
                                                      min_samples_split=5,
                                                      n_estimators=80,
                                                      random_state=0),
                 n_jobs=-1,
                 param_grid={'subsample': array([0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 ])},
                 scoring='precision')




```python
gsearch5.best_params_, gsearch5.best_score_
```




    ({'subsample': 0.8000000000000002}, 0.6587001254347038)




```python
sgb_tuned = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=14, min_samples_leaf=18, min_samples_split=5, max_features = 2, subsample=0.8, random_state=0)
sgb_tuned.fit(X, y)

sgb_prec_score = mean(get_score(sgb_tuned, X, y, 'precision'))
sgb_acc_score = mean(get_score(sgb_tuned, X, y, 'accuracy'))
print("Precision: {}, Accuracy: {}".format(sgb_prec_score, sgb_acc_score))
```

    Precision: 0.6587001254347037, Accuracy: 0.669128178913738
    


```python
sgb_fi = feat_importance(sgb_tuned, X)
sgb_fi[:10]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>max_standing</td>
      <td>0.329886</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Theatre_max</td>
      <td>0.321755</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U-Shaped_max</td>
      <td>0.155595</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Promoted / ticketed events</td>
      <td>0.066759</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Venue provides alcohol</td>
      <td>0.049638</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Loud music / events</td>
      <td>0.032210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wi-Fi</td>
      <td>0.024035</td>
    </tr>
    <tr>
      <th>3</th>
      <td>supervenue</td>
      <td>0.020122</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(sgb_fi)
```




    <AxesSubplot:ylabel='cols'>




    
![png](\images\portfolio\Wheelchair Ramp\output_112_1.png)
    



```python
result = pd.DataFrame({
    'Model': ['Fine-tuned Random Forest', 'Fine-tuned Stochastic Gradient Boosting'],
    'Precision Score': [rf_prec_score, sgb_prec_score],
    'Accuracy': [rf_acc_score,sgb_acc_score]})
    
result
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Precision Score</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fine-tuned Random Forest</td>
      <td>0.665685</td>
      <td>0.670403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fine-tuned Stochastic Gradient Boosting</td>
      <td>0.658700</td>
      <td>0.669128</td>
    </tr>
  </tbody>
</table>
</div>



It seems that the Random Forest model yields the highest precision and accuracy score so let's pick that as our final model.

## Final Evaluation
For the final evaluation we will use our fine-tuned Random Forest model to predict the test set we've set aside.


```python
df_test.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>venue_name</th>
      <th>Loud music / events</th>
      <th>Venue provides alcohol</th>
      <th>Wi-Fi</th>
      <th>supervenue</th>
      <th>U-Shaped_max</th>
      <th>max_standing</th>
      <th>Theatre_max</th>
      <th>Promoted / ticketed events</th>
      <th>Wheelchair accessible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3538</th>
      <td>the great hall and chambers leyton</td>
      <td>False</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>35.045455</td>
      <td>80</td>
      <td>112.715867</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>192</th>
      <td>dock street studios</td>
      <td>True</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>35.045455</td>
      <td>15</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2065</th>
      <td>clayton crown hotel</td>
      <td>False</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>80.000000</td>
      <td>380</td>
      <td>400.000000</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2490</th>
      <td>techspace aldgate east</td>
      <td>False</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>0</td>
      <td>112.715867</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>598</th>
      <td>the long acre</td>
      <td>True</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>35.045455</td>
      <td>200</td>
      <td>112.715867</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test = df_test.iloc[:, 1:-1]
y_test = df_test[target].replace({False: 1, True: 0})
```


```python
# Predict our test set using our trained Random Forest model.
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

rf_tuned = RandomForestClassifier(n_estimators= 300, min_samples_split= 25, min_samples_leaf = 2, max_features= 'sqrt', max_depth= 16, random_state=0)
rf_tuned.fit(X, y)

y_pred = rf_tuned.predict(X_test)
print("Precision:", precision_score(y_test, y_pred), ", Accuracy:", accuracy_score(y_test, y_pred))
```

    Precision: 0.6577669902912622 , Accuracy: 0.6636828644501279
    

As we can see the model achieve a pretty good performance on the test set with only a slight decrease from the training which is to be expected. This means that our model is able to generalize well in data it has not seen before. We are also able to achieve a precision of around 66% which satisfied one of this project initial requirement (ideally two-thirds of venues predicted to be without a ramp should not have a ramp).

## Outcome

- In conclusion we found out that features related to the capacity of the venue like `Theatre_max`, `max_standing`, and `U-Shaped_max` are an important predictor to determining whether a venue is wheelchair accessible, more importantly the larger the capacity the more likely they are to be wheelchair accessible. My hypothesis is that these venue that are larger must've have more budget and funding behind them, as such they are likely to be more well prepared and are able to afford wheelchair ramps to accomodate those with disability. Another feature that are also an important predictor is `promoted/ticketed` events which make sense since venue that host promoted/ticketed event would want to appease to all sort of audience in order to boost their income.

- After trying out different models I found out that decision tree based models works best to create a model that can predict whether or not a venue are wheel chair accessible. I then picked the two highest models - Random Forest and Stochastic Gradient Boosting, and perform hyperparameter tuning. In the end the Random Forest yields the better performance and so I picked it for the final evaluation. After evaluating on the unseen test set the model did a good job and yielded a precision and accuracy score of around 66%, this proves that the model doesn't only have high performance on the dataset it's train on but also on new unseen dataset which simulate real life application. 

- The evaluation on the test set also shows that the model have successfully achieve the ideal requirement of this project which is that ideally two-thirds of the venue predicted to be without a ramp doesn't actually have one.

## Future Works
- It is possible that some feature engineering on features that are related with the capacity of the venue like `Theatre_max`, `max_standing`, and `U-Shaped_max` might improve the performance of gradient descent based model (Logistic Regression) and distance based model (KNN) as they might be affected by multicollinearity.
