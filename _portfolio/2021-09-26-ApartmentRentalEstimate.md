---
title: Jakarta Apartment Rental Estimates
header:
  image: /portfolio/ApartmentHeader.png
date: 2021-09-26
tags:
  - Regression
  - Machine Learning
author_profile: true
toc: false
classes: wide
---

During the holiday I decided to make this regression project which try to estimate the prices of apartment rental in Jakarta. I know that there's a lot of these type of property prediction project out there, especially on Kaggle. So I try to make it different by scraping my own data from the Travelio website. Overall, this is a fun experience for me I got to learn a lot of new things like linear regression assumption, web scraping with Selenium and Beautiful Soup, geocoding with google Map API, and trying out various ensemble learning methods.

# Web Scraping
Since the code for the web scrapping is quite long I'll just attach a [link](https://github.com/DavidSamuell/Apartment-Rental-Price-Prediction-in-Jakarta/blob/main/travelioscrapper.ipynb) that leads to it. The scraping result is stored in the *travelio.csv* file.


```python
df = pd.read_csv('travelio.csv', index_col = 0)
```

# Data Cleaning


```python
print(df.isna().sum())
print(f"Number of column before cleaning: {df.shape[0]}")
```

    Name               0
    Total Bedroom      0
    Total Bathroom     0
    Apart Size         0
    Max Capacity      33
    Max Watt          28
    Address           33
    Swim Pool          0
    Rating             0
    Total Review      23
    Furnish Type       0
    Price              0
    dtype: int64
    Number of column before cleaning: 806
    

It seems that there are some rows that are missing some important informations. Since it will be really hard to fill in those data with appropriate value we will just remove them instead.


```python
# Remove rows that have no address
df2 = df.dropna(subset = ['Address'])

# Because our job is to predict apartment prices from the given specifications it wouldn't make sense to actually include 'Rating' and
# 'Total Review' since those variables can only be obtained when the apartment is already listed along with its price.
df2.drop(columns = ['Rating', 'Total Review'], inplace = True)
```


```python
print(df2.isna().sum())
print(f"Number of column after cleaning: {df2.shape[0]}")
```

    Name              0
    Total Bedroom     0
    Total Bathroom    0
    Apart Size        0
    Max Capacity      0
    Max Watt          0
    Address           0
    Swim Pool         0
    Furnish Type      0
    Price             0
    dtype: int64
    Number of column after cleaning: 773
    


```python
df2.reset_index(drop=True, inplace=True)
```


```python
# Clean the price column from the dataset and extract only the integer
df2['Price'] = df2['Price'].str.replace(r'\D+', '')
df2['Price'] = df2['Price'].astype(int)

df2['Apart Size'] = df2['Apart Size'].str.replace(r'\D+', '')
df2['Apart Size'] = df2['Apart Size'].astype(int)

df2['Max Capacity'] = df2['Max Capacity'].str.replace(r'\D+', '')
df2['Max Capacity'] = df2['Max Capacity'].astype(int)

df2['Max Watt'] = df2['Max Watt'].str.replace(r'\D+', '')
df2['Max Watt'] = df2['Max Watt'].astype(int)

df2['Total Bedroom'].replace({"Studio": 0}, inplace = True)
df2['Total Bedroom'] = df2['Total Bedroom'].astype(int)
```


```python
df2.info()
df2.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 773 entries, 0 to 772
    Data columns (total 10 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   Name            773 non-null    object
     1   Total Bedroom   773 non-null    int32 
     2   Total Bathroom  773 non-null    int64 
     3   Apart Size      773 non-null    int32 
     4   Max Capacity    773 non-null    int32 
     5   Max Watt        773 non-null    int32 
     6   Address         773 non-null    object
     7   Swim Pool       773 non-null    int64 
     8   Furnish Type    773 non-null    object
     9   Price           773 non-null    int32 
    dtypes: int32(5), int64(2), object(3)
    memory usage: 45.4+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Bedroom</th>
      <th>Total Bathroom</th>
      <th>Apart Size</th>
      <th>Max Capacity</th>
      <th>Max Watt</th>
      <th>Swim Pool</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>773.000000</td>
      <td>773.000000</td>
      <td>773.000000</td>
      <td>773.000000</td>
      <td>773.000000</td>
      <td>773.000000</td>
      <td>7.730000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.349288</td>
      <td>1.179819</td>
      <td>51.058215</td>
      <td>2.915912</td>
      <td>2545.725744</td>
      <td>0.878396</td>
      <td>6.816667e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.989382</td>
      <td>0.413511</td>
      <td>32.613103</td>
      <td>1.064947</td>
      <td>2016.470858</td>
      <td>0.478243</td>
      <td>4.224733e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>14.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>1.799999e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>2.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>4.199391e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>38.000000</td>
      <td>3.000000</td>
      <td>2200.000000</td>
      <td>1.000000</td>
      <td>5.281957e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>61.000000</td>
      <td>4.000000</td>
      <td>3500.000000</td>
      <td>1.000000</td>
      <td>8.206249e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>297.000000</td>
      <td>8.000000</td>
      <td>21000.000000</td>
      <td>1.000000</td>
      <td>3.777713e+07</td>
    </tr>
  </tbody>
</table>
</div>



# EDA


```python
sb.distplot(df2['Price'])
```




    <AxesSubplot:xlabel='Price', ylabel='Density'>




    
![png](\images\portfolio\Apartment Rental\output_11_1.png)
    



```python
print("Skewness: %f" % df2['Price'].skew())
print("Kurtosis: %f" % df2['Price'].kurt())

```

    Skewness: 1.993154
    Kurtosis: 5.708796
    

Skewness can indicate wether the data follow the normal distribution, in most cases a skewed data can actually lead to a worst performance of the model. Most model like linear regression works better with a data that follows the normal distribution. 

On the other hand, kurtosis indicate how heavy or light tailed the data is which is correlated to the presence of outliers. A kurtosis higher than 3 (normal distribution) normally indicate the presence of outliers and have a very peaked shape.


```python
sb.histplot(data = df2['Apart Size'] )
```




    <AxesSubplot:xlabel='Apart Size', ylabel='Count'>




    
![png](\images\portfolio\Apartment Rental\output_14_1.png)
    



```python
print("Skewness: %f" % df2['Apart Size'].skew())
print("Kurtosis: %f" % df2['Apart Size'].kurt())
```

    Skewness: 2.195653
    Kurtosis: 7.163900
    


```python
sb.pairplot(df2)
plt.rcParams["figure.figsize"] = [16,9]
plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_16_0.png)
    


Based on our scatter plot there is sign of outliers in Apart Size, we will investigate it further along the line.


```python
fig, ax = plt.subplots(figsize=(16, 9))
corr_matrix = df2.corr()
sb.heatmap(data = corr_matrix, annot = True)
plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_18_0.png)
    


The heatmap shows sign of multicollinearity between 'Total Bedroom', 'Total Bathroom', 'Apart Size', and 'Max Capacity'. We many need to remove some of these variables. The heatmap also shows a strong linear correlation between our target variable 'Price' and 'Apart Size' so its definitely a feature we want to keep.


```python
# Checkpoint
# df2.to_csv('travelio2.csv', index=True)
df2 = pd.read_csv('travelio2.csv', index_col = 0)
```

## Geocoding


```python
import geopy
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import GoogleV3
from geopy.extra.rate_limiter import RateLimiter

geolocator = GoogleV3(api_key = '***************')
```


```python
# Delay 1 second between each call to reduce the probability of a time out
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

df2['location'] = df2['Address'].apply(geocode)
```


```python
df2['point'] = df2["location"].apply(lambda loc: tuple(loc.point) if loc else None)
```

```python
# Unload the points to latitude and longtitude
df2[['lat', 'lon', 'altitude']] = pd.DataFrame(df2['point'].to_list(), index=df2.index)
```

Now that we have longitude and altitude of each address we don't need location and point features anymore. 

We should also remove altitude
because Google API require us to use the latitude api if we want to get latitude for a specific location. We won't go that far since its unlikely that altitude will play any significant roles on Jakarta apartments rental prices.


```python
df2.drop(columns = ['point', 'altitude'], inplace = True)
```


```python
# Checkpoint
# df2.to_csv('travelio3.csv', index=True)
df2 = pd.read_csv('travelio3.csv', index_col = 0)
```


```python
gmaps.configure(api_key='*********')

fig = gmaps.figure()
heatmap_layer = gmaps.heatmap_layer(
  df2[['lat','lon']],
  weights= df2['Price'],
  max_intensity = 100000000,
  point_radius= 10.0
)

fig.add_layer(heatmap_layer)
fig
```


![png](\images\portfolio\Apartment Rental\output_19_0.png)


Based on our geographical analysis we can see that apartment prices tend to be higher when located in Central and South Jakarta region. This is as expected as those 2 region are known as elite places in Jakarta with lots of facilities and well-developed infrastructures.

# Feature Engineering

## Multicollinearity

Since we are going to run a multiple linear regression model on this dataset it is important to avoid multicollinearity. When there is a correlation between the independent variables it can becomes difficult for the model to estimate the relationship between each independent variable and the dependent variable independently because the independent variables tend to change in unison.

Here are some algorithms that are effected by multicollinearity Linear Regression, Logistic Regression, KNN, and Naive Bayes.

It seems that 'Max Capacity' has a strong correlation with total Bedroom which indicate some multicollinearity. If we think about it, it makses sense because max capacity of an apartment is probably based on the number of bedroom and size of apartment.


```python
df2.drop(columns = "Max Capacity", inplace = True)
```


```python
X_variables = df2[['Total Bedroom', 'Total Bathroom', 'Apart Size', 'Max Watt']]

vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
```


```python
vif_data
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Total Bedroom</td>
      <td>4.468788</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Total Bathroom</td>
      <td>9.512638</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apart Size</td>
      <td>10.778528</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Max Watt</td>
      <td>3.888674</td>
    </tr>
  </tbody>
</table>
</div>



As seen on the heatmap and vif Total Bathroom and Apart Size seems highly correlated so we can remove one of them. I decided to remove Total Bathroom as it has a lower correlation with the target variable 'Price'.


```python
df2.drop(columns = "Total Bathroom", inplace = True)
```


```python
X_variables = df2[['Total Bedroom','Apart Size', 'Max Watt']]

vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
vif_data
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Total Bedroom</td>
      <td>4.280758</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apart Size</td>
      <td>6.205461</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Max Watt</td>
      <td>3.688159</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checkpoint
# df2.to_csv('travelio4.csv', index = True)
df2 = pd.read_csv('travelio4.csv', index_col = 0)
```

## Get Sub-Districts
I will try to obtain the list of districts and sub-districts each apartment is located in. This might help with our prediction since we now that some districts are more elite than others, these elite districts tend to have higher property prices.


```python
import requests
from bs4 import BeautifulSoup

URL = "https://id.wikipedia.org/wiki/Daftar_kecamatan_dan_kelurahan_di_Daerah_Khusus_Ibukota_Jakarta"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
table = soup.find_all("tbody")
```


```python
kecamatan = []
element = soup.select("td:nth-of-type(2)")
for idx, item in enumerate(element):
    if idx >5 and idx <48:
        kecamatan.append(item.text)
```


```python
old = ['Kelapa Gading', 'Pasar Minggu', 'Pasar Rebo', 'Tanjung Priok', 'Kebayoran Lama', 'Kebayoran Baru', 'Mampang Prapatan', 'Kebon Jeruk']
new = ['Klp. Gading', 'Ps. Minggu', 'Ps. Rebo', 'Tj. Priok', 'Kby. Lama', 'Kby. Baru','Mampang Prpt', 'Kb. Jeruk']
for i, val in enumerate(old):
    idx = kecamatan.index(val)
    kecamatan[idx] = new[i]
kecamatan.append('Setia Budi')
kecamatan.append('Pd. Aren')
kecamatan.append('Kebon Jeruk')
```


```python
# Create a function that will return the district an address belongs in, and apply that function to every row in the data
def kec(val):
    return next((x for x in kecamatan if x.lower() in val.lower()), None)
df2['kecamatan'] = df2['location'].apply(kec)
```


```python
a = df2[df2['kecamatan'].isnull()].index.tolist()
empty_rows = df2.iloc[a].copy()
```


```python
print(a)
```
    [96, 97, 110, 164, 172, 190, 233, 235, 240, 243, 249, 573, 627, 641, 646, 668, 682, 688, 689, 743]

It seems that some locations are incomplete (they don't have district in them) due to the geocoding by Google API, so we will just use our original Address that we obtain from the travelio website instead.


```python
rev = testdf.index.to_list()

def kec(val):
    return next((x for x in kecamatan if x.lower() in val.lower()), None)
df2['kecamatan'].iloc[rev] = df2['Address'].iloc[rev].apply(kec)

df2.replace({'Setia Budi' : 'Setiabudi'}, inplace=True)
df2.replace({'Kebon Jeruk' : 'Kb. Jeruk'}, inplace=True)
```


```python
a= df2[df2['kecamatan'].isnull()].index.tolist()
len(df2.iloc[a])
```




    0



Great! We've filled in all the districts for each corresponding apartment. Now lets try to see the average price of housing for each district.


```python
df2.groupby(['kecamatan']).Price.agg('mean').sort_values(ascending = False)
# res = df_agg.apply(lambda x: x.sort_values(ascending=True))
```




    kecamatan
    Kby. Baru            1.806751e+07
    Mampang Prpt         1.523673e+07
    Setiabudi            1.106564e+07
    Kby. Lama            1.089827e+07
    Menteng              9.563951e+06
    Cilandak             8.927184e+06
    Pancoran             8.576855e+06
    Tanah Abang          7.904778e+06
    Pd. Aren             7.735000e+06
    Gambir               6.446666e+06
    Kb. Jeruk            6.430905e+06
    Pesanggrahan         6.270067e+06
    Tebet                5.843505e+06
    Grogol Petamburan    5.749463e+06
    Senen                5.660513e+06
    Kembangan            5.596787e+06
    Kemayoran            5.463723e+06
    Tambora              5.285560e+06
    Kalideres            5.280624e+06
    Ps. Minggu           5.273301e+06
    Pademangan           5.096884e+06
    Kramat Jati          4.981074e+06
    Penjaringan          4.865202e+06
    Jatinegara           4.780318e+06
    Klp. Gading          4.778076e+06
    Sawah Besar          4.761854e+06
    Ps. Rebo             4.500187e+06
    Taman Sari           4.375700e+06
    Cengkareng           4.273109e+06
    Cempaka Putih        4.173814e+06
    Pulo Gadung          3.681220e+06
    Tj. Priok            3.512187e+06
    Duren Sawit          2.959583e+06
    Cakung               2.623194e+06
    Name: Price, dtype: float64



As we can see, the top 5 districts with the highest apartment price in average comes from district located in South and Central Jakarta. This make sense since we know that places in South and Central Jakarta have a higher living cost and are more elite compare to other region.

## A bit more cleaning

While I was extracting district for each row I've also discovered that some of the Max Watt have only a value of 1, it seems that I forgot to clean these rows. These 1's indicate that the scraper are unable to extract the max watt information during the web scraping process.

So, I'll have to perform some imputing. To impute this data I've decided to use multivariate imputing from Sk-Learn which uses a Bayesian Ridge Regression to predict the most likely outcome of the missing data using other features in the data. We basically treat Max Watt as the dependent variables.

This should result to a more accurate imputation compared to the univariate imputation in which only a single value obtain by calculating the mean or median is use to fill in all the missing data.


```python
# Checkpoints
# df2.to_csv('travelio5.csv', index = True)
df2 = pd.read_csv('travelio5.csv', index_col=0)
```


```python
df2['Max Watt'] = df2['Max Watt'].replace(1, np.NaN)
df2['Max Watt'].isna().sum()
```




    66




```python
# Here I created a temporary dataframe and fill it with features that have highest correlations with Max Watt.
# These features will act as the independent/predictors variable to determine the most likely value for Max Watt.

# I pick these features as predictor because they have the highest correlation with Max Watt
dftemp = df2[['Total Bedroom', 'Apart Size', 'Max Watt', 'Price']].copy()
```


```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imputer = IterativeImputer()
imputer.fit(dftemp)
df_trans = imputer.transform(dftemp)
df_trans = pd.DataFrame(df_trans, columns=dftemp.columns)
```


```python
df2['Max Watt'] = df_trans['Max Watt']
df2['Max Watt'] = df2['Max Watt'].astype(int)
```


```python
# Checkpoints
# df2.to_csv('travelio6.csv', index = True)
df2 = pd.read_csv('travelio6.csv', index_col = 0)
```

## Outliers


```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x = df2['Apart Size'], y = df2['Price'])
plt.ylabel('Price', fontsize=13)
plt.xlabel('Apart Size', fontsize=13)
plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_68_0.png)
    


We can see that there is one apartment with a very high price despite its fairly average size, we can assume that it is located in a very elite location but unfortunately is simply way to high and unlikely in a real life scenario so we can simply treat it as an outlier and remove that data point.


```python
# Delete Outliers
df2 = df2.drop(df2[df2['Price'] > 35000000].index)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x = df2['Apart Size'], y = df2['Price'])
plt.ylabel('Price', fontsize=13)
plt.xlabel('Apart Size', fontsize=13)
plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_70_0.png)
    


## Handling Categorical Data


```python
df2.reset_index(drop=True, inplace=True)
```


```python
# Clean some row
df2['Furnish Type'] = df2['Furnish Type'].replace('5', "Unfurnished")
```


```python
# Dummy Encoding Furnish Type because it is a nominal variable
dummy_furnished = pd.get_dummies(df2[['Furnish Type', 'kecamatan']], prefix='', prefix_sep='')
df2 = pd.merge(
    left=df2,
    right=dummy_furnished,
    left_index=True,
    right_index=True,
)
df2.drop(columns = ["Furnish Type", 'kecamatan'], inplace = True)
```

## Normality

To check for the distribution and normality of our features I'll be plotting a distribution plot and a normal probability plot / Q-Q plot.

Features with normal distribution should have a symemtrical bell shape curve in the distribution plot and data distribution should closely follow the diagonal in the Q-Q plot.


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
sb.distplot(df2['Price'], fit=norm, ax=ax1)
res = stats.probplot(df2['Price'], plot=ax2)
```


    
![png](\images\portfolio\Apartment Rental\output_77_0.png)
    


After removing the outliers in 'Sales Price' we can see that the kurtosis becomes closer to that of a normal distribution. However we can still see some skewness in our data so we can fix it using box cox transformation. We will also apply it to "apart size".


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
sb.distplot(df2['Apart Size'], fit=norm, ax=ax1)
res = stats.probplot(df2['Apart Size'], plot=ax2)
```


    
![png](\images\portfolio\Apartment Rental\output_79_0.png)
    


We can analyze that the Apartment rental pricing are skewed to the left and has a heavy tail distribution.

### Log Transform

To fix the skewness and transform our data into a more normal distribution I'm going to perform a log transformation. The reason I did not go with box cox transformation is because it will be difficult to reverse transform our price estimate during the final production.


```python
df2['Apart Size'] = np.log(df2['Apart Size'])
df2['Price'] = np.log(df2['Price'])
```


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
sb.distplot(df2['Price'], fit=norm, ax=ax1)
res = stats.probplot(df2['Price'], plot=ax2)
```


    
![png](\images\portfolio\Apartment Rental\output_84_0.png)
    



```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
sb.distplot(df2['Apart Size'], fit=norm, ax=ax1)
res = stats.probplot(df2['Apart Size'], plot=ax2)
```


    
![png](\images\portfolio\Apartment Rental\output_85_0.png)
    


## Testing Homoscedasticity

Homoscedasticity refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)'. Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.

The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).


```python
plt.subplots(figsize=(8, 6))
plt.scatter(df2['Price'], df2['Apart Size'])
```




    <matplotlib.collections.PathCollection at 0x27361563e80>




    
![png](\images\portfolio\Apartment Rental\output_88_1.png)
    


As you can see the scatter plot doesn't have a conic shape anymore. Thats the power of normality. Just by ensuring normality in some variables, we can ensure homoscedasticity.


```python
fig, ax = plt.subplots(figsize=(16, 9))
corr_matrix = df2.iloc[:, :12].corr()
sb.heatmap(data = corr_matrix, annot = True)
plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_90_0.png)
    



```python
# Checkpoint
# df2.to_csv('travelio7.csv', index=True)
df2 = pd.read_csv('travelio7.csv', index_col=0)
```

## Feature Selection


```python
df2.head()
X_var = df2.drop(columns= ['Name', 'lat', 'lon', 'Swim Pool', 'location', 'Address', 'Price'])
y = df2['Price']

X_var.describe()
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Bedroom</th>
      <th>Apart Size</th>
      <th>Max Watt</th>
      <th>Full Furnished</th>
      <th>Unfurnished</th>
      <th>Cakung</th>
      <th>Cempaka Putih</th>
      <th>Cengkareng</th>
      <th>Cilandak</th>
      <th>Duren Sawit</th>
      <th>...</th>
      <th>Ps. Rebo</th>
      <th>Pulo Gadung</th>
      <th>Sawah Besar</th>
      <th>Senen</th>
      <th>Setiabudi</th>
      <th>Taman Sari</th>
      <th>Tambora</th>
      <th>Tanah Abang</th>
      <th>Tebet</th>
      <th>Tj. Priok</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>...</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
      <td>772.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.347150</td>
      <td>3.780989</td>
      <td>2796.141192</td>
      <td>0.922280</td>
      <td>0.077720</td>
      <td>0.007772</td>
      <td>0.007772</td>
      <td>0.025907</td>
      <td>0.009067</td>
      <td>0.003886</td>
      <td>...</td>
      <td>0.005181</td>
      <td>0.069948</td>
      <td>0.023316</td>
      <td>0.036269</td>
      <td>0.081606</td>
      <td>0.006477</td>
      <td>0.003886</td>
      <td>0.068653</td>
      <td>0.012953</td>
      <td>0.012953</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.988234</td>
      <td>0.519492</td>
      <td>1881.756917</td>
      <td>0.267904</td>
      <td>0.267904</td>
      <td>0.087873</td>
      <td>0.087873</td>
      <td>0.158960</td>
      <td>0.094851</td>
      <td>0.062257</td>
      <td>...</td>
      <td>0.071841</td>
      <td>0.255225</td>
      <td>0.151003</td>
      <td>0.187081</td>
      <td>0.273941</td>
      <td>0.080269</td>
      <td>0.062257</td>
      <td>0.253027</td>
      <td>0.113147</td>
      <td>0.113147</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>2.639057</td>
      <td>900.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>3.433987</td>
      <td>1917.250000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <th>50%</th>
      <td>2.000000</td>
      <td>3.637586</td>
      <td>2200.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <th>75%</th>
      <td>2.000000</td>
      <td>4.110874</td>
      <td>3500.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <th>max</th>
      <td>3.000000</td>
      <td>5.693732</td>
      <td>21000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
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
<p>8 rows × 39 columns</p>
</div>



## Train Test Split

I'm going to split the data into a test set and a training set. I will hold out the test set until the very end and use the error on those data as an unbiased estimate of how my models did.

I might perform a further split later on the training set into training set proper and a validation set or I might cross-validate.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_var, y, test_size=0.2, random_state=27)
```

# 2. Model

## 2.0 Cross-Validation Routine


```python
kf = KFold(n_splits=5, shuffle=True, random_state=27)
```


```python
from sklearn.model_selection import KFold

# squared_loss
def rmse_cv(model):
    rmse = -cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv = kf)
    return(rmse)
```


```python
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0), n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=27)
    if axes is None:
        _, axes = plt.subplots(figsize=(10, 8))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return plt
```


```python
def train_model(title, estimator):  
    cv = rmse_cv(estimator)
    cv_error = cv.mean()
    cv_std = cv.std()
    # fit
    estimator.fit(X_train, y_train)
    # predict
    y_train_pred = estimator.predict(X_train)
    training_error = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    y_test_pred = estimator.predict(X_test)
    test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # visualizing the result
    df = pd.DataFrame({'Model':title, '(RMSE) CV Error': cv_error, 'CV Std': cv_std, '(RMSE) Training error':training_error, '(RMSE) Test Error':test_error, '(R2) Training Score':train_r2, '(R2) Test Score':test_r2}, index=[0])
    return df
```

## 2.1 Linear Regression


```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr_results = train_model('Linear Regression', lr)
lr_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0.229457</td>
      <td>0.016247</td>
      <td>0.209983</td>
      <td>0.217076</td>
      <td>0.835861</td>
      <td>0.817331</td>
    </tr>
  </tbody>
</table>
</div>




```python
title = "Learning Curves"

estimator = lr
plot_learning_curve(estimator, title, X_var, y)

plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_105_0.png)
    


## 2.2 Random Forest Regressor


```python
rf = RandomForestRegressor(max_depth = 6, random_state = 27)
rf_results = train_model('Random Forest Regressor [baseline]', rf)
rf_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest Regressor [baseline]</td>
      <td>0.24443</td>
      <td>0.015315</td>
      <td>0.202519</td>
      <td>0.234036</td>
      <td>0.847323</td>
      <td>0.787672</td>
    </tr>
  </tbody>
</table>
</div>



### Hyperparameter Tuning

We will try to perfrom some hyperparameter tuning to see if we can improve our score on the test set


```python
n_estimators = [int(x) for x in np.linspace(start = 100, stop=2000, num=20)]
max_features = [None, 'sqrt', 'log2']
max_depth = [np.arange(start = 3, stop = 16, step=1)]
max_depth.append(None)
min_samples_split = [2, 4, 5, 6, 7]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid ={'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator=rf, cv = 5, param_distributions = random_grid, random_state=27, n_jobs=-1, verbose=0)
rf_random.fit(X_train, y_train)
rf_random.best_params_
```




    {'n_estimators': 1700,
     'min_samples_split': 7,
     'min_samples_leaf': 1,
     'max_features': 'sqrt',
     'max_depth': None,
     'bootstrap': False}




```python
best_random = rf_random.best_estimator_
rf_best_results = train_model('Random Forest [optimized]', best_random)
rf_best_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest [optimized]</td>
      <td>0.217775</td>
      <td>0.009606</td>
      <td>0.103904</td>
      <td>0.203305</td>
      <td>0.959811</td>
      <td>0.839772</td>
    </tr>
  </tbody>
</table>
</div>




```python
title = "Learning Curves"

estimator = best_random
plot_learning_curve(estimator, title, X_var, y)

plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_112_0.png)
    


If we take a look at the curve we can see that the trend shows an increasing validation score, if we have more data we might be able to get it close to the training score. This is one of the downside of working with limited data on ensemble model.

## 2.3 XGBoost Regressor


```python
xgb = XGBRegressor(verbosity = 0, random_state = 27)
xgb_results = train_model('XGBRegressor [baseline]', xgb)
xgb_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XGBRegressor [baseline]</td>
      <td>0.224384</td>
      <td>0.009016</td>
      <td>0.071544</td>
      <td>0.228214</td>
      <td>0.980946</td>
      <td>0.798104</td>
    </tr>
  </tbody>
</table>
</div>




```python
d = {'Learning Rate':[],
            'Mean CV Error': [],
            'CV Error Std': [],
            'Training Error': []}
for lr in np.arange(start= 0.05, stop=0.35, step=0.05):
    xgb = XGBRegressor(n_estimators=500, learning_rate=lr, early_stopping=5)
    cv_res = rmse_cv(xgb)
    xgb.fit(X_train, y_train)
    y_train_pred = xgb.predict(X_train)
    d['Learning Rate'].append(lr)
    d['Mean CV Error'].append(cv_res.mean())
    d['CV Error Std'].append(cv_res.std())
    d['Training Error'].append(np.sqrt(mean_squared_error(y_train, y_train_pred)))

xgb_tuning_1 = pd.DataFrame(d)
xgb_tuning_1
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Learning Rate</th>
      <th>Mean CV Error</th>
      <th>CV Error Std</th>
      <th>Training Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.05</td>
      <td>0.216038</td>
      <td>0.011774</td>
      <td>0.092352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.10</td>
      <td>0.225486</td>
      <td>0.007819</td>
      <td>0.058462</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.15</td>
      <td>0.228645</td>
      <td>0.005337</td>
      <td>0.052823</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.20</td>
      <td>0.227991</td>
      <td>0.003771</td>
      <td>0.051646</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.25</td>
      <td>0.228967</td>
      <td>0.006944</td>
      <td>0.051581</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.30</td>
      <td>0.231707</td>
      <td>0.005858</td>
      <td>0.051467</td>
    </tr>
  </tbody>
</table>
</div>




```python
xgb_tuning_2 = pd.DataFrame(d)
xgb_tuning_2
print('Optimal parameter values are: ')
best = xgb_tuning_2.iloc[xgb_tuning_2.idxmin()['Mean CV Error']]
print('max_depth: {}'.format(int(best['max_depth'])))
print('min_child_weight: {}'.format(int(best['min_child_weight'])))
```

    Optimal parameter values are: 
    max_depth: 3
    min_child_weight: 1
    


```python
d = {'max_depth':[],
             'min_child_weight': [],
            'Mean CV Error': [],
            'CV Error Std': [],
            'Training Error': []}

params2 = {'max_depth': list(range(3,10,2)), 'min_child_weight': list(range(1,6,2))}

for md in params2['max_depth']:
    for mcw in params2['min_child_weight']:
        xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping=5, max_depth=md, min_child_weight=mcw )
        cv_res = rmse_cv(xgb_model)
        xgb_model.fit(X_train, y_train)
        y_train_xgb = xgb_model.predict(X_train)
        d['max_depth'].append(md)
        d['min_child_weight'].append(mcw)
        d['Mean CV Error'].append(cv_res.mean())
        d['CV Error Std'].append(cv_res.std())
        d['Training Error'].append(np.sqrt(mean_squared_error(y_train_xgb, y_train)))
```


```python
n_estimators = np.arange(500, 1100, 100)
learning_rate = np.arange(0.05, 0.35, 0.05)
max_depth = np.arange(3,10,2)
min_child_weight = np.arange(1,6,2)

param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth = max_depth, min_child_weight = min_child_weight)

xgb_random = RandomizedSearchCV(estimator=xgb, cv = kf, param_distributions = param_grid, scoring = 'neg_root_mean_squared_error', random_state=27, n_jobs=-1)
xgb_random.fit(X_train, y_train)
xgb_random.best_params_
```




    {'n_estimators': 800,
     'min_child_weight': 1,
     'max_depth': 3,
     'learning_rate': 0.05}




```python
xgb_best = xgb_random.best_estimator_
xgb_best_results = train_model('XGBRegressor [optimized]', xgb_best)
xgb_best_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XGBRegressor [optimized]</td>
      <td>0.208752</td>
      <td>0.017149</td>
      <td>0.133361</td>
      <td>0.205716</td>
      <td>0.933794</td>
      <td>0.835949</td>
    </tr>
  </tbody>
</table>
</div>




```python
title = "Learning Curves"

estimator = xgb_best
plot_learning_curve(estimator, title, X_var, y)

plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_121_0.png)
    


Based on the curve trend we can see that there is still potential for the training and validation/test score to improve given more data just like our Random Forest.

## 2.4 Gradient Boosting


```python
gb = GradientBoostingRegressor(random_state = 27) 
gbr_results = train_model('Gradient Boosting', gb)
gbr_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gradient Boosting</td>
      <td>0.220045</td>
      <td>0.013427</td>
      <td>0.172792</td>
      <td>0.21274</td>
      <td>0.888855</td>
      <td>0.824556</td>
    </tr>
  </tbody>
</table>
</div>



### Hyperparameter Tuning


```python
learning_rate = np.arange(start= 0.05, stop=0.2, step=0.01)
n_estimators = np.arange(start = 100, stop = 2050, step=50)
max_depth = [x for x in np.linspace(1, 10, num=10)]
min_samples_split = [2, 4, 5, 6, 7]
min_samples_leaf = [1, 2, 4]
max_features = [None, 'sqrt', 'log2']



random_grid ={}

random_grid = {
                'learning_rate':learning_rate,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features,
                'min_samples_split':min_samples_split,
                'min_samples_leaf':min_samples_leaf
              }
gb_random = RandomizedSearchCV(estimator=gb, cv = kf, param_distributions = random_grid, scoring = 'neg_root_mean_squared_error', random_state=27, n_jobs=-1)
gb_random.fit(X_train, y_train)
gb_random.best_params_
```




    {'n_estimators': 650,
     'min_samples_split': 6,
     'min_samples_leaf': 1,
     'max_features': 'log2',
     'max_depth': 4.0,
     'learning_rate': 0.07}




```python
gb_best = gb_random.best_estimator_
gb_best_results = train_model('Gradient Boosting [optimized]', gb_best)
gb_best_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gradient Boosting [optimized]</td>
      <td>0.215909</td>
      <td>0.013687</td>
      <td>0.104709</td>
      <td>0.197649</td>
      <td>0.959186</td>
      <td>0.848563</td>
    </tr>
  </tbody>
</table>
</div>




```python
title = "Learning Curves"

estimator = gb_best
plot_learning_curve(estimator, title, X_var, y)

plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_128_0.png)
    


Similar to other ensemble model learning curve we can see an increasing trend in our validation score, given more data the accuracy might improve even more. But. even so this model yielded the highest accuracy so far.

## 2.5 Ridge Regression


```python
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = Ridge(random_state = 27)

ridge_rob = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

ridge_rob_results = train_model('Ridge Regression (rob)', ridge_rob)
ridge_rob_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ridge Regression (rob)</td>
      <td>0.230955</td>
      <td>0.016572</td>
      <td>0.211195</td>
      <td>0.216493</td>
      <td>0.833961</td>
      <td>0.81831</td>
    </tr>
  </tbody>
</table>
</div>




```python
ridge_norm = make_pipeline(MinMaxScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
ridge_norm_results = train_model('Ridge Regression (norm)', ridge_norm)
ridge_norm_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ridge Regression (norm)</td>
      <td>0.229402</td>
      <td>0.015968</td>
      <td>0.210444</td>
      <td>0.216502</td>
      <td>0.835139</td>
      <td>0.818296</td>
    </tr>
  </tbody>
</table>
</div>




```python
ridge_std = make_pipeline(StandardScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
ridge_std_results = train_model('Ridge Regression (std)', ridge_std)
ridge_std_results
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ridge Regression (std)</td>
      <td>0.229708</td>
      <td>0.016488</td>
      <td>0.21007</td>
      <td>0.217024</td>
      <td>0.835726</td>
      <td>0.817418</td>
    </tr>
  </tbody>
</table>
</div>




```python
title = "Learning Curves"

estimator = ridge_rob
plot_learning_curve(estimator, title, X_var, y)
    
plt.show()
```


    
![png](\images\portfolio\Apartment Rental\output_134_0.png)
    



```python
pd.concat([lr_results, rf_best_results, xgb_best_results, gb_best_results, ridge_rob_results], axis=0, ignore_index=True)
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
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>(RMSE) CV Error</th>
      <th>CV Std</th>
      <th>(RMSE) Training error</th>
      <th>(RMSE) Test Error</th>
      <th>(R2) Training Score</th>
      <th>(R2) Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0.229457</td>
      <td>0.016247</td>
      <td>0.209983</td>
      <td>0.217076</td>
      <td>0.835861</td>
      <td>0.817331</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest [optimized]</td>
      <td>0.217775</td>
      <td>0.009606</td>
      <td>0.103904</td>
      <td>0.203305</td>
      <td>0.959811</td>
      <td>0.839772</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBRegressor [optimized]</td>
      <td>0.208752</td>
      <td>0.017149</td>
      <td>0.133361</td>
      <td>0.205716</td>
      <td>0.933794</td>
      <td>0.835949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gradient Boosting [optimized]</td>
      <td>0.215909</td>
      <td>0.013687</td>
      <td>0.104709</td>
      <td>0.197649</td>
      <td>0.959186</td>
      <td>0.848563</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ridge Regression (rob)</td>
      <td>0.230955</td>
      <td>0.016572</td>
      <td>0.211195</td>
      <td>0.216493</td>
      <td>0.833961</td>
      <td>0.818310</td>
    </tr>
  </tbody>
</table>
</div>



Comparing all these different models we can see that **XGBRegressor** showed the best result with a CV error of 0.208752, the reason I picked this model even though **Random Forest** and **Gradient Boosting** showed a better test score and error is because we don't want to select model based on our test performance, if we did that it would mean that our model overfit the test data since it may have a better performance due to chance alone. It's best practice to perform any sort of model selection based on our CV error since it is more representative of our model generalization capability.

The performance on the test set should only then be use as a true out of sample performance metric once we've selected our model. Based on the difference in training and test performance we can also see that there is a high variance in our model due to the lack of data, this is further supported by the trend on the learning curve where the test score are still increasing alongside with the number of training data.

# Exporting Model

```python
import pickle
with open('travelio_apart.pickle','wb') as f:
    pickle.dump(xgb_best,f)
```


```python
import json
columns = {
    'data_columns' : [col.lower() for col in X_var.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
```
