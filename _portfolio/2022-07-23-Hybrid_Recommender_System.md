---
title: Building a Hybrid Recommendation System for Amazon
excerpt: "<img src='/images/portfolio/AmazonThumbnail.jpg'>
<br><br>
In this project I'll be creating a hybrid collaborative-filtering content-based recommendation system for Amazon Product using Tensorflow Keras API.
<br><br>
<b>Tags: Deep Learning | Recommendation System | Tensorflow Keras | Word2Vec</b>"
date: 2022-07-23
tags:
  - Deep Learning
  - Recommendation System
  - Word2Vec
  - Data Science
collection: portfolio
---

<br/>
# Introduction
Most of us are familiar with Recommendation System, almost every major tech comany has applied them in some way or another. Amazon uses it to suggest products to customers, YouTube uses it to decide which video to play next on autoplay, and companies like Netflix and Spotify depend highly on the effectiveness of their recommendation engines for their business and success. 

Recommendation system provides recommendation by predicting the rating or preference that user would give to an item. In this project I will be creating a hybrid content-based & collaborative filtering recommendation system on the Electronic Category subset of the [Amazon Product Review Dataset](https://nijianmo.github.io/amazon/index.html).


<br/> Before we dive into the code let's learn some terminology.

**Cold-Start Problem**

Imagine opening a new brand new app and there aren't any subscribers yet. This is where the cold-start problem appears, basically there are no recorded user or product interactions yet so it will be hard for a recommended systems to detect any pattern and make recommendations.

In order to solve this problem we can apply a **Knowledge-Based approach**: for example, asking for user’s preferences in order to create an initial profile, or using demographic information (i.e. high school shows for teenagers and cartoons for kids). Then once we have an idea of what items does the user like we can use a Content-Based filtering to recommend item with similar features.
<br/><br/>

**Popularity Bias**

The popularity bias is a phenomenon in recommendation algorithms where popular items tend to be suggested over products that sell in small quantities, even if the latter would be of reasonable interest for individuals.<br/><br/>

**Content-Based**

Content-Based methods are based on the product contents. For instance, if User A likes Product 1, and Product 2 is similar to Product 1, then User A would probably like Product 2 as well. Two products are similar if they have similar features.

In a nutshell, the idea is that users actually rate the features of the product and not the product itself. To put it in another way, if I like products related to music and art, it’s because I like those features (music and art). Based on that, we can estimate how much I would like other products with the same features. This method is best suited for situations where there are known data on products but not on users.

Content-Based recommendation system are robust towards the **popularity bias ** and **cold-start problem** because they don't rely on the interactions between users and products like Collaborative Filtering does. The downside is that it can only recommend items with features similar to the original item. This limits the scope of recommendations, and can also result in surfacing items with low ratings.

*Reference: https://towardsdatascience.com/modern-recommendation-systems-with-neural-networks-3cc06a6ded2c*
<br/><br/>

**Collaborative-Filtering**

Collaborative Filtering is based on the assumption that similar users like similar products. For instance, if User A likes Product 1, and User B is similar to User A, then User B would probably like Product 1 as well. Two users are similar if they like similar products.

This method doesn’t need product features to work, it requires many ratings from many users instead. Because of this it is prone to the cold-start problem where a new user doesn't have any ratings yet. It is also prone to the popularity bias where it tends to only recommend items that are popular since popular items have lots of interactions with a lot's of user, so it is difficult for collaborative filters to accurately recommend novel or niche items. 

The upside however is that it is always “self-generating” — users create the data for you naturally as they interact with items. This can be a valuable data source, especially in cases where high-quality item features are not available or difficult to obtain. Another benefit of collaborative filters is that it helps users discover new items that are outside the subspace defined by their historical profile.

*Reference: https://towardsdatascience.com/creating-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-cc8b431618af*

<br/> **Hybrid Model**

Hybrid recommender system is a special type of recommender system that combines both content and collaborative filtering method. Combining collaborative filtering and content-based filtering could be more effective in some cases. Hybrid approaches can be implemented in several ways: by making content-based and collaborative-based predictions separately and then combining them; by adding content-based capabilities to a collaborative-based approach (and vice versa). Several studies empirically compare the performance of the hybrid with pure collaborative and content-based methods and demonstrate that hybrid methods can provide more accurate recommendations than pure approaches. These methods can also be used to overcome some of the problems that the individuals content-based or collaborative filtering approach have such as the cold start and the popularity bias problem.

<br/> **Embedding**

Embeddings are an important concept in collaborative filtering, formally they can be defined as a multi-dimensional vector representation of a particular entity. Embeddings represents a set of parameters/features that will represent the characteristic of each product and user, so for example whether a user like a particular genre of movie or if a movie is filled with action. Sometimes these parameters are also called as *latent factors*. You can imagine that each users and products have their own embedding/latent factors in the shape of a n-dimensional vector together they make what is known as an embedding matrix.

So how do we actually find the value for these embeddings or paremeters? The answer is, we don't. We will let our model learn them. By analyzing the existing relations between users and movies, our model can figure out the best value for the emebddings. 

First we will attribute to each of our users and each of our products a random vector (embedding) of a certain length, and we will make those learnable parameters. That means that at each step, when we compute the loss by comparing our predictions (dot product of user and product embedding vector) to our targets (rating of a product given by that user), we will compute the gradients of the loss with respect to those embedding vectors and update them with the rules of SGD (or another optimizer). At the beginning, those numbers don't mean anything since we have chosen them randomly, but by the end of training, they will.


```python
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Reshape, Dot, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from urllib.request import urlopen

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
import gzip
import pandas as pd
```


```python
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Appliances.csv
!wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Appliances.json.gz # Appliances metadata
```

    --2022-07-18 04:54:18--  http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Appliances.csv
    Resolving deepyeti.ucsd.edu (deepyeti.ucsd.edu)... 169.228.63.50
    Connecting to deepyeti.ucsd.edu (deepyeti.ucsd.edu)|169.228.63.50|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 25158234 (24M) [application/octet-stream]
    Saving to: ‘Appliances.csv’
    
    Appliances.csv      100%[===================>]  23.99M  9.49MB/s    in 2.5s    
    
    2022-07-18 04:54:21 (9.49 MB/s) - ‘Appliances.csv’ saved [25158234/25158234]
    
    --2022-07-18 04:54:21--  http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Appliances.json.gz
    Resolving deepyeti.ucsd.edu (deepyeti.ucsd.edu)... 169.228.63.50
    Connecting to deepyeti.ucsd.edu (deepyeti.ucsd.edu)|169.228.63.50|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 59884788 (57M) [application/octet-stream]
    Saving to: ‘meta_Appliances.json.gz’
    
    meta_Appliances.jso 100%[===================>]  57.11M  16.7MB/s    in 4.4s    
    
    2022-07-18 04:54:25 (13.1 MB/s) - ‘meta_Appliances.json.gz’ saved [59884788/59884788]
    
    


```python
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Electronics.csv
```

    --2022-07-22 15:00:33--  http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Electronics.csv
    Resolving deepyeti.ucsd.edu (deepyeti.ucsd.edu)... 169.228.63.50
    Connecting to deepyeti.ucsd.edu (deepyeti.ucsd.edu)|169.228.63.50|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 876247561 (836M) [application/octet-stream]
    Saving to: ‘Electronics.csv’
    
    Electronics.csv     100%[===================>] 835.65M  33.8MB/s    in 25s     
    
    2022-07-22 15:00:59 (33.4 MB/s) - ‘Electronics.csv’ saved [876247561/876247561]
    
    

# Collaborative Filtering

## Load Product Ratings


```python
df = pd.read_csv('Electronics.csv', names = ['ProductID', 'UserID', 'Rating', 'Timestamp'])
```

## Filter Sparse Products and Users

First up since we have about 20 million rows on our dataset our RAM won't be able to handle the processing and training so let's reduce it by filtering some of the sparse data, more specifically we'll only take products and users that have more than 10 ratings.


```python
# Filter sparse products
min_product_ratings = 10
filter_products = (df['ProductID'].value_counts() > min_product_ratings)
filter_products = filter_products[filter_products].index.tolist()

# Filter sparse users
min_user_ratings = 10
filter_users = (df['UserID'].value_counts() > min_user_ratings)
filter_users = filter_users[filter_users].index.tolist()

# Actual filtering
df = df[(df['ProductID'].isin(filter_products)) & (df['UserID'].isin(filter_users))]
print('Shape User-Ratings after filtering:\t{}'.format(df.shape))
# print('Shape User-Ratings filtered:\t{}'.format(df_filterd.shape))
```

    Shape User-Ratings after filtering:	(3307989, 4)
    

As we can see we end up with about 3.3 million data which should be plenty enough


```python
df.head()
```





  <div id="df-1cdfc1ab-4006-45cd-bcd4-a3457a79dbd5">
    <div class="colab-df-container">
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
      <th>ProductID</th>
      <th>UserID</th>
      <th>Rating</th>
      <th>Timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>0060786817</td>
      <td>A2BZ7MYTSNYMEW</td>
      <td>4.0</td>
      <td>1154304000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0060786817</td>
      <td>A2GQ0WGM9BYX9O</td>
      <td>5.0</td>
      <td>1145577600</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0151004714</td>
      <td>A1ER5AYS3FQ9O3</td>
      <td>5.0</td>
      <td>1220313600</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0312171048</td>
      <td>AQ2UN38CMK119</td>
      <td>5.0</td>
      <td>1262217600</td>
    </tr>
    <tr>
      <th>144</th>
      <td>0373250460</td>
      <td>A3U5CZ3PV82JXD</td>
      <td>1.0</td>
      <td>1115078400</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1cdfc1ab-4006-45cd-bcd4-a3457a79dbd5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1cdfc1ab-4006-45cd-bcd4-a3457a79dbd5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1cdfc1ab-4006-45cd-bcd4-a3457a79dbd5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We can see that our ProductID and UserID is in the form of a string, we want to convert it to an integer so that our model can process it. We also want to make sure that these integer are ordered from 0 up to the number of users/products - 1. The reason is that these integer will be use to index the embedding matrix  which maps each user and each movie to an embedding vector.

The code below does precisely that, we first convert the Product and User ID column into a categorical object then we call `cat.codes` which returns a series containing the integer representation of each unique ID.


```python
df.UserID = pd.Categorical(df.UserID)
df['NewUserID'] = df.UserID.cat.codes

df.ProductID = pd.Categorical(df.ProductID)
df['NewProductID'] = df.ProductID.cat.codes
```


```python
df.head()
```





  <div id="df-f7bcf22a-0c08-4306-b932-29d780541006">
    <div class="colab-df-container">
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
      <th>ProductID</th>
      <th>UserID</th>
      <th>Rating</th>
      <th>Timestamp</th>
      <th>NewUserID</th>
      <th>NewProductID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>0060786817</td>
      <td>A2BZ7MYTSNYMEW</td>
      <td>4.0</td>
      <td>1154304000</td>
      <td>67323</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0060786817</td>
      <td>A2GQ0WGM9BYX9O</td>
      <td>5.0</td>
      <td>1145577600</td>
      <td>73918</td>
      <td>1</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0151004714</td>
      <td>A1ER5AYS3FQ9O3</td>
      <td>5.0</td>
      <td>1220313600</td>
      <td>20897</td>
      <td>4</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0312171048</td>
      <td>AQ2UN38CMK119</td>
      <td>5.0</td>
      <td>1262217600</td>
      <td>176874</td>
      <td>5</td>
    </tr>
    <tr>
      <th>144</th>
      <td>0373250460</td>
      <td>A3U5CZ3PV82JXD</td>
      <td>1.0</td>
      <td>1115078400</td>
      <td>143243</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f7bcf22a-0c08-4306-b932-29d780541006')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f7bcf22a-0c08-4306-b932-29d780541006 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f7bcf22a-0c08-4306-b932-29d780541006');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Next we're going to split the data. We're going to use sklearn `train_test_split` which shuffle the data and split it according to the specified size in our case its 70% training and 30% test set. 


```python
train, test = train_test_split(df, test_size=0.3, random_state=0)
```


```python
train_user = train["NewUserID"].values
train_product = train["NewProductID"].values
train_ratings = train["Rating"].values

test_user = test["NewUserID"].values
test_product = test["NewProductID"].values
test_ratings = test["Rating"].values
```

## Matrix Factorization

For our first model we are going to try collaborative filtering using matrix factorization. In matrix fatorization we take the dot product of each user and product embedding vector to get prediction of the rating a user will give to a particular product.


```python
# Make a neural network

# Get number of users and number of movies
N = df["NewUserID"].nunique()
M = df["NewProductID"].nunique()

# Set embedding dimension
K = 10

# User input
u = Input(shape=(1,))

# Product input
p = Input(shape=(1,))

# User enmbedding
u_emb = Embedding(N, K)(u) #output is (num_samples, 1, K)

# Product embedding
p_emb = Embedding(M, K)(p) #output is (num_samples, 1, K)

# Flatten both embeddings
u_emb = Flatten()(u_emb) # now it's (num_samples, K)
p_emb = Flatten()(p_emb) # now it's (num_samples, K)

# Concatenate user and movie embeddings into a feature vector
output = Dot(1, normalize = False)([u_emb, p_emb]) # now it's (num_samples, 2K)
```


```python
# Build the model and compile
model1 = Model(inputs=[u, p], outputs=output)
model1.compile(
    loss='mse',
    optimizer=SGD(lr=0.1, momentum = 0.99), 
)
```

    /usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/gradient_descent.py:102: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    


```python
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)
```


```python
r = model1.fit(
    x = [train_user, train_product],
    y = train_ratings,
    epochs =  50,
    batch_size = 1024,
    validation_data = ([test_user, test_product], test_ratings),
    callbacks=[es])
```

    Epoch 1/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 19.9108 - val_loss: 19.9073
    Epoch 2/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 19.9037 - val_loss: 19.9017
    Epoch 3/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 19.8544 - val_loss: 19.8314
    Epoch 4/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 19.4423 - val_loss: 19.2368
    Epoch 5/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 17.9577 - val_loss: 17.3993
    Epoch 6/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 15.2774 - val_loss: 14.6737
    Epoch 7/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 12.3012 - val_loss: 12.0183
    Epoch 8/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 9.7677 - val_loss: 9.8333
    Epoch 9/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 7.8363 - val_loss: 8.1710
    Epoch 10/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 6.4294 - val_loss: 6.9532
    Epoch 11/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 5.4133 - val_loss: 6.0632
    Epoch 12/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 4.6602 - val_loss: 5.3976
    Epoch 13/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 4.0837 - val_loss: 4.8937
    Epoch 14/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 3.6306 - val_loss: 4.5075
    Epoch 15/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 3.2669 - val_loss: 4.2143
    Epoch 16/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 2.9704 - val_loss: 3.9728
    Epoch 17/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 2.7188 - val_loss: 3.7828
    Epoch 18/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 2.5048 - val_loss: 3.6212
    Epoch 19/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 2.3204 - val_loss: 3.4875
    Epoch 20/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 2.1590 - val_loss: 3.3789
    Epoch 21/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 2.0164 - val_loss: 3.2896
    Epoch 22/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.8880 - val_loss: 3.2091
    Epoch 23/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.7697 - val_loss: 3.1392
    Epoch 24/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.6613 - val_loss: 3.0844
    Epoch 25/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.5611 - val_loss: 3.0305
    Epoch 26/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.4669 - val_loss: 2.9863
    Epoch 27/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.3788 - val_loss: 2.9440
    Epoch 28/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.2969 - val_loss: 2.9049
    Epoch 29/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.2204 - val_loss: 2.8689
    Epoch 30/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.1492 - val_loss: 2.8350
    Epoch 31/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.0821 - val_loss: 2.8036
    Epoch 32/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 1.0198 - val_loss: 2.7753
    Epoch 33/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.9623 - val_loss: 2.7466
    Epoch 34/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.9077 - val_loss: 2.7186
    Epoch 35/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.8573 - val_loss: 2.6921
    Epoch 36/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.8099 - val_loss: 2.6675
    Epoch 37/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.7660 - val_loss: 2.6436
    Epoch 38/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.7249 - val_loss: 2.6203
    Epoch 39/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.6869 - val_loss: 2.5995
    Epoch 40/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.6512 - val_loss: 2.5778
    Epoch 41/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.6177 - val_loss: 2.5592
    Epoch 42/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.5863 - val_loss: 2.5427
    Epoch 43/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.5567 - val_loss: 2.5260
    Epoch 44/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.5288 - val_loss: 2.5098
    Epoch 45/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.5026 - val_loss: 2.4936
    Epoch 46/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.4780 - val_loss: 2.4798
    Epoch 47/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.4553 - val_loss: 2.4693
    Epoch 48/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.4338 - val_loss: 2.4581
    Epoch 49/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.4139 - val_loss: 2.4469
    Epoch 50/50
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.3952 - val_loss: 2.4384
    


```python
r = model1.fit(
    x = [train_user, train_product],
    y = train_ratings,
    epochs =  20,
    batch_size = 1024,
    validation_data = ([test_user, test_product], test_ratings),
    callbacks=[es])
```

    Epoch 1/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.3779 - val_loss: 2.4292
    Epoch 2/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.3617 - val_loss: 2.4260
    Epoch 3/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.3465 - val_loss: 2.4177
    Epoch 4/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.3323 - val_loss: 2.4135
    Epoch 5/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.3189 - val_loss: 2.4099
    Epoch 6/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.3063 - val_loss: 2.4065
    Epoch 7/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2945 - val_loss: 2.4026
    Epoch 8/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2835 - val_loss: 2.4007
    Epoch 9/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2730 - val_loss: 2.3989
    Epoch 10/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2632 - val_loss: 2.3968
    Epoch 11/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2541 - val_loss: 2.3971
    Epoch 12/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2454 - val_loss: 2.3959
    Epoch 13/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2373 - val_loss: 2.3957
    Epoch 14/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2296 - val_loss: 2.3967
    Epoch 15/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2225 - val_loss: 2.3972
    Epoch 16/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2158 - val_loss: 2.3996
    Epoch 17/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2094 - val_loss: 2.3994
    Epoch 18/20
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.2034 - val_loss: 2.4013
    Epoch 19/20
    2241/2262 [============================>.] - ETA: 0s - loss: 0.1978Restoring model weights from the end of the best epoch: 13.
    2262/2262 [==============================] - 7s 3ms/step - loss: 0.1979 - val_loss: 2.4041
    Epoch 19: early stopping
    


```python
from sklearn.metrics import mean_squared_error
y_pred = model1.predict([test_user, test_product])
y_true = test_ratings

#  Compute RMSE
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))
```

    
    
    Testing Result With Keras Matrix-Factorization: 1.5478 RMSE
    

Not bad we got an RMSE of 1.5478 by training our model for 63 epoch with an SGD optimizer, I set the learning rate to 0.1 and momentum to 0.9 as it seems to yield the best result, I've also tried using an Adam Optimizer but it takes much longer to train and the loss seems to be really large. 

Let's see if we can improve the performance with a modified version of matrix factorization.

## Deep Learning
This approach is similar to the matrix factorization above, but instead of using a fixed dot-product to get our ratings we will concatenate the embedding vector of each user and product and then feed them as input into a dense layers. The output will come out of a dense layer with one node which will become the predicted ratings. By utilizing the dense layer our neural network can hopefully find better combinations of value for each embeddings vectors that minimize the loss.


```python
# Make a neural network

# Get number of users and number of movies
N = df["NewUserID"].nunique()
M = df["NewProductID"].nunique()

# Set embedding dimension
K = 32

# User input
u = Input(shape=(1,))

# Product input
p = Input(shape=(1,))

# User enmbedding
u_emb = Embedding(N, K)(u) #output is (num_samples, 1, K)

# Product embedding
p_emb = Embedding(M, K)(p) #output is (num_samples, 1, K)

# Flatten both embeddings
u_emb = Flatten()(u_emb) # now it's (num_samples, K)
p_emb = Flatten()(p_emb) # now it's (num_samples, K)

# Concatenate user and movie embeddings into a feature vector
x = Concatenate()([u_emb, p_emb]) # now it's (num_samples, 2K)

# Now that we have a feature vector, it's just a regular ANN
dense = Dense(1024, activation='relu')(x)
dense = Dropout(0.2)(dense)

output = Dense(1)(dense)
```


```python
# Build the model and compile
model2 = Model(inputs=[u, p], outputs=output)
model2.compile(
    loss='mse',
    optimizer=SGD(lr=0.1, momentum = 0.9),
)
```

    /usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/gradient_descent.py:102: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    


```python
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    verbose=1,
    restore_best_weights=True
)
```


```python
# # Batch Size of 1024
r2 = model2.fit(
    x = [train_user, train_product],
    y = train_ratings,
    epochs = 20,
    batch_size = 1024,
    verbose = 2,
    validation_data = ([test_user, test_product], test_ratings),
    callbacks=[es]
)
```

    Epoch 1/20
    2262/2262 - 13s - loss: 1.3648 - val_loss: 1.3383 - 13s/epoch - 6ms/step
    Epoch 2/20
    2262/2262 - 8s - loss: 1.3370 - val_loss: 1.3387 - 8s/epoch - 4ms/step
    Epoch 3/20
    2262/2262 - 8s - loss: 1.3370 - val_loss: 1.3383 - 8s/epoch - 4ms/step
    Epoch 4/20
    2262/2262 - 8s - loss: 1.3367 - val_loss: 1.3414 - 8s/epoch - 4ms/step
    Epoch 5/20
    2262/2262 - 8s - loss: 1.3369 - val_loss: 1.3385 - 8s/epoch - 4ms/step
    Epoch 6/20
    2262/2262 - 8s - loss: 1.3368 - val_loss: 1.3384 - 8s/epoch - 4ms/step
    Epoch 7/20
    Restoring model weights from the end of the best epoch: 1.
    2262/2262 - 8s - loss: 1.3369 - val_loss: 1.3384 - 8s/epoch - 4ms/step
    Epoch 7: early stopping
    


```python
# Batch Size of 512
r2 = model2.fit(
    x = [train_user, train_product],
    y = train_ratings,
    epochs = 20,
    batch_size = 512,
    verbose = 2,
    validation_data = ([test_user, test_product], test_ratings),
    callbacks=[es]
)
```

    Epoch 1/20
    4523/4523 - 19s - loss: 1.3517 - val_loss: 1.3383 - 19s/epoch - 4ms/step
    Epoch 2/20
    4523/4523 - 17s - loss: 1.3379 - val_loss: 1.3385 - 17s/epoch - 4ms/step
    Epoch 3/20
    4523/4523 - 17s - loss: 1.3382 - val_loss: 1.3502 - 17s/epoch - 4ms/step
    Epoch 4/20
    4523/4523 - 16s - loss: 1.3381 - val_loss: 1.3409 - 16s/epoch - 4ms/step
    Epoch 5/20
    4523/4523 - 16s - loss: 1.3381 - val_loss: 1.3415 - 16s/epoch - 4ms/step
    Epoch 6/20
    4523/4523 - 16s - loss: 1.3210 - val_loss: 1.2964 - 16s/epoch - 4ms/step
    Epoch 7/20
    4523/4523 - 16s - loss: 1.2586 - val_loss: 1.2103 - 16s/epoch - 4ms/step
    Epoch 8/20
    4523/4523 - 17s - loss: 1.1305 - val_loss: 1.1344 - 17s/epoch - 4ms/step
    Epoch 9/20
    4523/4523 - 17s - loss: 1.0250 - val_loss: 1.1279 - 17s/epoch - 4ms/step
    Epoch 10/20
    4523/4523 - 17s - loss: 0.9662 - val_loss: 1.1442 - 17s/epoch - 4ms/step
    Epoch 11/20
    4523/4523 - 17s - loss: 0.9291 - val_loss: 1.1440 - 17s/epoch - 4ms/step
    Epoch 12/20
    4523/4523 - 17s - loss: 0.8992 - val_loss: 1.1387 - 17s/epoch - 4ms/step
    Epoch 13/20
    4523/4523 - 17s - loss: 0.8706 - val_loss: 1.1291 - 17s/epoch - 4ms/step
    Epoch 14/20
    4523/4523 - 17s - loss: 0.8400 - val_loss: 1.1715 - 17s/epoch - 4ms/step
    Epoch 15/20
    Restoring model weights from the end of the best epoch: 9.
    4523/4523 - 17s - loss: 0.8092 - val_loss: 1.2015 - 17s/epoch - 4ms/step
    Epoch 15: early stopping
    


```python
# plot losses
plt.plot(r2.history['loss'], label = 'train loss')
plt.plot(r2.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()
```


    
![png](\images\portfolio\RecommendationSystem\output_33_0.png)
    



```python
from sklearn.metrics import mean_squared_error
y_pred = model2.predict([test_user, test_product])
y_true = test_ratings

#  Compute RMSE
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))
```

    
    
    Testing Result With Keras Matrix-Factorization: 1.0620 RMSE
    

Amazing by modifying our Matrix Factorization model with a simple dense and dropout layer we manage to improve our model performance evident by the decrease in RMSE loss by 0.4858. Also for this model I use the same hyperparameter as before except for the batch size as using a lower batch size of 512 actually trains the model much better compare to the 1024 batch size.

Last but not least let us try to build a hybrid collaborative-content model.

# Hybrid Content-Collaborative Recommendation System

## Load and Clean Product Metadata


```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from matplotlib import pyplot
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    


```python
col_list = ['title', 'feature', 'description', 'category', 'brand', 'asin']
```

You can find the metadata for Electronic Product ratings in the following link: http://deepyeti.ucsd.edu/jianmo/amazon/index.html

Keep in find that it's a json.gz file so we'll need to parse it first using the json module, unfortunately due to the very large file it is impossible to parse it without running out of RAM in Google Collab. So I downloaded and parse it on my local machine and convert it into a csv.gz file and upload it to my Drive


```python
meta_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/metadata/meta_Electronics.csv.gz', compression = 'gzip', usecols = col_list)
```


```python
meta_df.head()
```





  <div id="df-af4eb374-0f12-4037-b281-c886ed88af56">
    <div class="colab-df-container">
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
      <th>category</th>
      <th>description</th>
      <th>title</th>
      <th>brand</th>
      <th>feature</th>
      <th>asin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>['Electronics', 'Camera &amp;amp; Photo', 'Video S...</td>
      <td>The following camera brands and models have be...</td>
      <td>Genuine Geovision 1 Channel 3rd Party NVR IP S...</td>
      <td>GeoVision</td>
      <td>['Genuine Geovision 1 Channel NVR IP Software'...</td>
      <td>0011300000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>['Electronics', 'Camera &amp;amp; Photo']</td>
      <td>This second edition of the Handbook of Astrono...</td>
      <td>Books "Handbook of Astronomical Image Processi...</td>
      <td>33 Books Co.</td>
      <td>['Detailed chapters cover these fundamental to...</td>
      <td>0043396828</td>
    </tr>
    <tr>
      <th>2</th>
      <td>['Electronics', 'eBook Readers &amp;amp; Accessori...</td>
      <td>A zesty tale. (Publishers Weekly)&lt;br /&gt;&lt;br /&gt;G...</td>
      <td>One Hot Summer</td>
      <td>Visit Amazon's Carolina Garcia Aguilera Page</td>
      <td>[]</td>
      <td>0060009810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>['Electronics', 'eBook Readers &amp; Accessories',...</td>
      <td>NaN</td>
      <td>Hurray for Hattie Rabbit: Story and pictures (...</td>
      <td>Visit Amazon's Dick Gackenbach Page</td>
      <td>[]</td>
      <td>0060219602</td>
    </tr>
    <tr>
      <th>4</th>
      <td>['Electronics', 'eBook Readers &amp; Accessories',...</td>
      <td>&amp;#8220;sex.lies.murder.fame. is brillllli&amp;#821...</td>
      <td>sex.lies.murder.fame.: A Novel</td>
      <td>Visit Amazon's Lolita Files Page</td>
      <td>[]</td>
      <td>0060786817</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-af4eb374-0f12-4037-b281-c886ed88af56')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-af4eb374-0f12-4037-b281-c886ed88af56 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-af4eb374-0f12-4037-b281-c886ed88af56');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Okay so to implement the content-based aspect to our hybrid model we are going to need something that represent the feature of each product. Fortunately the metadata provide us with the feature column which describe the features of each product. 

However there is a lot of empty and dirty text in the feature column so we'll have to perform some text cleaning. First up let's remove the bracket and convert all the blank text into null value.


```python
#Remove square bracket
meta_df['feature'] = meta_df['feature'].str.strip('[]')
meta_df['feature'] = meta_df['feature'].replace(r'^\s*$', np.nan, regex=True)
```


```python
meta_df['feature'].isna().sum()
```




    137120



Up next let's fill in the missing value with value from the description column


```python
meta_df['feature'].fillna(meta_df['description'], inplace = True)
```


```python
meta_df['feature'].isna().sum()
```




    81948



Okay it seems that there are still a lot of missing value, so let's use the product category to fill in the missing value.


```python
meta_df['feature'].fillna(meta_df['category'], inplace = True)
```


```python
meta_df['feature'].isna().sum()
```




    0



Up next we are going to clean the text in the feature column by removing ASCII characters, stop words, html, punctuation, and converting them to lower case.


```python
#Utitlity functions for removing ASCII characters, converting lower case, removing stop words, html and punctuation.

def _removeNonAscii(s):
    return "".join(i for i in s if ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_numerical(text):
  return ''.join([i for i in s if not i.isdigit()])

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

meta_df['clean_feature'] = meta_df['feature'].apply(remove_html)
meta_df['clean_feature'] = meta_df.clean_feature.apply(make_lower_case)
meta_df['clean_feature'] = meta_df.clean_feature.apply(remove_stop_words)
meta_df['clean_feature'] = meta_df.clean_feature.apply(_removeNonAscii)
meta_df['clean_feature'] = meta_df.clean_feature.apply(remove_punctuation)
```

Now let's check if there are any blank text after cleaning unwanted text from our feature column.


```python
# Convert string with empty spaces into NaN
meta_df['clean_feature'] = meta_df['clean_feature'].replace(r'^\s*$', np.nan, regex=True)
```


```python
meta_df[meta_df['clean_feature'].isna()]
```





  <div id="df-bcdd20c6-52c0-41c0-bacb-fd7777d6cc38">
    <div class="colab-df-container">
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
      <th>category</th>
      <th>description</th>
      <th>title</th>
      <th>brand</th>
      <th>feature</th>
      <th>asin</th>
      <th>clean_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>['Electronics', 'Computers &amp;amp; Accessories',...</td>
      <td>&lt;b&gt; &lt;/b&gt;</td>
      <td>My MacBook</td>
      <td>Visit Amazon's John Ray Page</td>
      <td>&lt;b&gt; &lt;/b&gt;</td>
      <td>0789743035</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>994</th>
      <td>['Electronics', 'Computers &amp;amp; Accessories',...</td>
      <td>.</td>
      <td>//</td>
      <td>.</td>
      <td>'`'</td>
      <td>9579215065</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1269</th>
      <td>['Electronics', 'Portable Audio &amp; Video', 'Por...</td>
      <td>.</td>
      <td>Digilife Fold-Up Compact Speakers for Apple iP...</td>
      <td>DekCell</td>
      <td>.</td>
      <td>9864216155</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1839</th>
      <td>['Electronics', 'Camera &amp; Photo', 'Accessories...</td>
      <td>&lt;style type="text/css"&gt; &lt;!-- .style1 {color: #...</td>
      <td>Premium 6 ft Panasonic RP-CDHM15-K Mini 1.3c H...</td>
      <td>A Days Tech</td>
      <td>&lt;style type="text/css"&gt; &lt;!-- .style1 {color: #...</td>
      <td>9981739367</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1872</th>
      <td>['Electronics', 'Accessories &amp; Supplies', 'Aud...</td>
      <td>&lt;style type="text/css"&gt; &lt;!-- .style1 {color: #...</td>
      <td>Premium Canon Mini HDMI cable 10 feet</td>
      <td>LASUS</td>
      <td>&lt;style type="text/css"&gt; &lt;!-- .style1 {color: #...</td>
      <td>9983891212</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>745875</th>
      <td>['Electronics', 'Security &amp; Surveillance', 'Ho...</td>
      <td>&lt;div class="boost-aplus-container"&gt; &lt;div class...</td>
      <td>Halo+ Smart Smoke and CO Alarm plus Weather Al...</td>
      <td>Halo: Safety Reimagined</td>
      <td>&lt;div class="boost-aplus-container"&gt; &lt;div class...</td>
      <td>B01D4X1QSO</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>762170</th>
      <td>['Electronics', 'Computers &amp; Accessories', 'Co...</td>
      <td>??</td>
      <td>Sunburst USB C to USB C (2M Black)</td>
      <td>Sunburst Worldwide</td>
      <td>'??'</td>
      <td>B01ETLETXO</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>764235</th>
      <td>['Electronics', 'Portable Audio &amp; Video', 'MP3...</td>
      <td>d</td>
      <td>iPod Touch 6 Case, MagicMobile Hybrid Rugged S...</td>
      <td>Emopeak</td>
      <td>d</td>
      <td>B01F2J8ASM</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>771317</th>
      <td>['Electronics', 'Television &amp; Video']</td>
      <td>NaN</td>
      <td>New Small 7 Tech Media Box Gen 3 IPTV Chinese ...</td>
      <td>Small 7 Tech</td>
      <td>''</td>
      <td>B01FUHDPIQ</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>772244</th>
      <td>['Electronics', 'Computers &amp; Accessories', 'Ne...</td>
      <td>....</td>
      <td>Sabrent 7-Port USB 3.0 Hub with Individual Pow...</td>
      <td>Sabrent</td>
      <td>'.', '.', '.', '.'</td>
      <td>B01FXE650Q</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>591 rows × 7 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bcdd20c6-52c0-41c0-bacb-fd7777d6cc38')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bcdd20c6-52c0-41c0-bacb-fd7777d6cc38 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bcdd20c6-52c0-41c0-bacb-fd7777d6cc38');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Okay as we can see we get blank text in our clean_feature column due to gibberish text from the feature or description column, to fix this let's fill in our clean_feature column with the product category.


```python
meta_df['clean_feature'].fillna(meta_df['category'], inplace = True)
```

Don't forget that we need to remove the punctuations so let's' clean the column once again.


```python
meta_df['clean_feature'] = meta_df.clean_feature.apply(func = make_lower_case)
meta_df['clean_feature'] = meta_df.clean_feature.apply(func = remove_stop_words)
meta_df['clean_feature'] = meta_df.clean_feature.apply(func = remove_punctuation)
```


```python
meta_df['clean_feature'] = meta_df['clean_feature'].replace(r'^\s*$', np.nan, regex=True)
```


```python
meta_df['clean_feature'].fillna(meta_df['category'], inplace = True)
```


```python
meta_df[meta_df['clean_feature'].isna()]
```





  <div id="df-19b43167-b426-48ed-a749-75e606068e0b">
    <div class="colab-df-container">
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
      <th>category</th>
      <th>description</th>
      <th>title</th>
      <th>brand</th>
      <th>feature</th>
      <th>asin</th>
      <th>clean_feature</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-19b43167-b426-48ed-a749-75e606068e0b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-19b43167-b426-48ed-a749-75e606068e0b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-19b43167-b426-48ed-a749-75e606068e0b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# Checkpoint
# meta_df.to_csv('/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/metadata/meta_df.csv.gz', compression='gzip', index = False)
```


```python
clean_meta_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/metadata/clean_meta_df.csv.gz', compression='gzip')
```

Okay great now that we have a clean text describing the features of each product we can just feed in into our neural network right? Well wrong, we have to remember that neural network can only work with numbers and not string. So what we have to do is find a way to represent these words with numerical value. Fortunately we can do so with **Word Embeddings**.

**Word Embeddings**

Word Embeddings are dense, low dimension numerical vector that represents a word and are able to capture the semantic meaning of the word very well. One of the more popular method to obtain these word embeddings is the **Word2Vec**. It was introduced in 2013 by Google and has been shown to outperformed the TF-IDF method, for an intuitive explanation of how it works you can visit the following [article](https://jalammar.github.io/illustrated-word2vec/) by Jay Allamar. But to simplify Word2Vec takes a word and returns a vector in D-dimensional space which we called as word embeddings. Training our own word embeddings is an expensive process and also requires a large dataset, so it is common practice to use a pre-trained word embeddings. 

I'll be using Google pre-trained word embeddings which contains 300-dimensional vectors for 3 million words and phrases, these embeddings are obtained through training on the Google News dataset which consists of around 100 billion words. You can obtain them from the following link: https://code.google.com/archive/p/word2vec/

**Average Word2Vec**

Okay now the question is how do we convert each row in the clean_feature column which consist of a bunch of words (sentences) into one word embeddings? There are multiple approaches out there but I'm going to go with the average Word2Vec approach. What it does is first we split the sentences into words and find the vector representation or word embeddings for each word. Then we will sum all of the word vectors and divide the sum by the total number of words in the sentence, very simple!

Some of the other approach you could try is the TF-IDF Word2Vec which takes the sum of the word vectors multiplied by the TF-IDF score for each word and divide it by the total sum of the TF-IDF vectors. Unfortunately this approach takes a really long time to run and can take days to complete with a large dataset like the one we're dealing with here.


## Average Word2Vec


```python
corpus = []
for words in clean_meta_df['clean_feature']:
    corpus.append(words.split())
```


```python
EMBEDDING_FILE = '/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/GoogleNews-vectors-negative300.bin'
google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Training our corpus with Google Pretrained Model

google_model = Word2Vec(size = 300, window=5, min_count = 2, workers = -1)
google_model.build_vocab(corpus)

google_model.intersect_word2vec_format(EMBEDDING_FILE, lockf=1.0, binary=True)

google_model.train(corpus, total_examples=google_model.corpus_count, epochs = 5)
```


```python
# Function to perform average Word2Vec
def vectors(df):
    # Creating a list for storing the vectors (description into vectors)
    global word_embeddings
    word_embeddings = []

    # Reading the each book description 
    for line in df['clean_feature']:
        avgword2vec = None
        count = 0
        for word in line.split():
            if word in google_model.wv.vocab:
                count += 1
                if avgword2vec is None:
                    avgword2vec = google_model[word]
                else:
                    avgword2vec = avgword2vec + google_model[word]
                
        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
            word_embeddings.append(np.array(avgword2vec))
        else:
            word_embeddings.append(np.nan)
```


```python
vectors(clean_meta_df)
```


```python
np_word_embeddings = np.array(word_embeddings)
```


```python
# Checkpoint
# np.save('/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/word_vector/avg_word_vector.npy', np_word_embeddings)
```


```python
# Empty Features Replaced with Description then Replaced with Categories
avg_word_vector = np.load("/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/word_vector/avg_word_vector.npy", allow_pickle = True)
```


```python
clean_meta_df.rename(columns={'asin':'ProductID'}, inplace = True)
```


```python
clean_meta_df['feature_embedding'] = [x for x in avg_word_vector]
```

Here I perform some further cleaning 


```python
clean_meta_df.feature_embedding.isna().sum()
```




    8




```python
clean_meta_df = clean_meta_df[clean_meta_df['feature_embedding'].notna()]
```


```python
clean_meta_df.feature_embedding.isna().sum()
```




    0



## Merge Ratings with Metadata

Okay now that we have the word embeddings for our product `feature` column let's merge it with the original rating dataset.


```python
len(df)
```




    6275164




```python
merge_df = df.merge(clean_meta_df.drop_duplicates(subset=['ProductID']), how = 'left')
```


```python
merge_df.isna().sum()
```




    ProductID                 0
    UserID                    0
    Rating                    0
    Timestamp                 0
    category               3130
    description          538072
    title                  3140
    brand                  6856
    feature                3130
    clean_feature          3130
    feature_embedding      3130
    dtype: int64



It seems that there are some products with no available metadata, let's remove these product ratings. We'll start by removing products that have no available title.


```python
# Remove rows where title is empty
merge_df = merge_df[merge_df['title'].notna()]
```


```python
merge_df.isna().sum()
```




    ProductID                 0
    UserID                    0
    Rating                    0
    Timestamp                 0
    category                  0
    description          534937
    title                     0
    brand                  3726
    feature                   0
    clean_feature             0
    feature_embedding         0
    dtype: int64




```python
print(f'rows removed: {len(df)-len(merge_df)}')
```

    rows removed: 3140
    

Now that we've succesfully remove all the the products with no metadata, let's remove some of the columns we won't be using to save up some memory. After that we'll just repeat the process we've done before such as mapping the User and Product ID to an integer from 0 - size of the Users/Products, split the dataset. 


```python
merge_df.drop(columns=['description', 'feature', 'clean_feature'], inplace = True)
```


```python
merge_df.UserID = pd.Categorical(merge_df.UserID)
merge_df['NewUserID'] = merge_df.UserID.cat.codes

merge_df.ProductID = pd.Categorical(merge_df.ProductID)
merge_df['NewProductID'] = merge_df.ProductID.cat.codes
```


```python
from sklearn.model_selection import train_test_split
 
train, test = train_test_split(merge_df, test_size=0.3,random_state=10)
```

One additional thing that we need to do here is to prepare the word embeddings so that they are in the right format which is a 2 dimension array. To do that we can just loop through the `word_embeddings` column and convert each row to a numpy array using list comprehension and finally storing all of it in a numpy array.


```python
train_user = train["NewUserID"].values
train_product = train["NewProductID"].values
train_ratings = train["Rating"].values
train_feature_emb = np.array([x for x in np.array(train['feature_embedding'])])

test_user = test["NewUserID"].values
test_product = test["NewProductID"].values
test_ratings = test["Rating"].values
test_feature_emb = np.array([x for x in np.array(test['feature_embedding'])])
```

## Deep Hybrid System With Metadata And Keras

So to actually implement the word embeddings into our collaborative model and turn it into a hybrid model is to just concatenate thema long with our User and Product embedding. And that's it we then simply feed it into a Dense layer and then add a Dropout layer as we did before and we've created a hybrid model.


```python
# Make a neural network

# Get number of users and number of movies
N = len(set(merge_df["NewUserID"]))
M = len(set(merge_df["NewProductID"]))

# Set embedding dimension
K = 32

# User input
u = Input(shape=(1,))

# Product input
p = Input(shape=(1,))

# Description Word Vector
feature_word_vector = Input(shape=[train_feature_emb.shape[1]])

# User enmbedding
u_emb = Embedding(N, K)(u) #output is (num_samples, 1, K)

# Product embedding
p_emb = Embedding(M, K)(p) #output is (num_samples, 1, K)

# Flatten both embeddings
u_emb = Flatten()(u_emb) # now it's (num_samples, K)
p_emb = Flatten()(p_emb) # now it's (num_samples, K)

# Concatenate user and movie embeddings, and the description word vector into a feature vector
x = Concatenate()([u_emb, p_emb, feature_word_vector]) 

# Now that we have a feature vector, it's just a regular ANN
dense = Dense(1024, activation='relu')(x)
dense = Dropout(0.2)(dense)

output = Dense(1)(dense)
```


```python
checkpoint_filepath = "/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/model(v2)"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    monitor='val_loss',
    verbose = 0,
    save_best_only=True,
)
```


```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=6,
    verbose=1,
    restore_best_weights=True
)
```

I've tried a lot of different hyperparameters such as a lower learning rate, adam optimizer, and lower batch size, but I found that a high learning rate of 0.1 with momentum of 0.9 and batch_size of 1024 works best for this hybrid model on this specific dataset. It yielded the lowest loss just a few  epoch.


```python
# Build the model and compile
hybrid_model = Model(inputs=[u, p, feature_word_vector], outputs=output)
hybrid_model.compile(
    loss='mse',
    optimizer=SGD(lr=0.1, momentum = 0.9),
)
```

    /usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/gradient_descent.py:102: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    


```python
r3 = hybrid_model.fit(
    x = [train_user, train_product, train_feature_emb],
    y = train_ratings,
    epochs = 20,
    batch_size = 1024,
    verbose = 2,
    callbacks=[model_checkpoint, early_stopping],
    validation_data = ([test_user, test_product, test_feature_emb], test_ratings)
)
```

    Epoch 1/20
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/model(v2)/assets
    2260/2260 - 13s - loss: 1.3298 - val_loss: 1.3226 - 13s/epoch - 6ms/step
    Epoch 2/20
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/model(v2)/assets
    2260/2260 - 9s - loss: 1.2499 - val_loss: 1.2078 - 9s/epoch - 4ms/step
    Epoch 3/20
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/model(v2)/assets
    2260/2260 - 9s - loss: 1.1042 - val_loss: 1.1320 - 9s/epoch - 4ms/step
    Epoch 4/20
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/model(v2)/assets
    2260/2260 - 9s - loss: 0.9824 - val_loss: 1.1083 - 9s/epoch - 4ms/step
    Epoch 5/20
    2260/2260 - 8s - loss: 0.9322 - val_loss: 1.1242 - 8s/epoch - 4ms/step
    Epoch 6/20
    2260/2260 - 8s - loss: 0.9040 - val_loss: 1.1261 - 8s/epoch - 4ms/step
    Epoch 7/20
    2260/2260 - 8s - loss: 0.8826 - val_loss: 1.1357 - 8s/epoch - 4ms/step
    Epoch 8/20
    2260/2260 - 8s - loss: 0.8630 - val_loss: 1.1533 - 8s/epoch - 4ms/step
    Epoch 9/20
    2260/2260 - 8s - loss: 0.8416 - val_loss: 1.1552 - 8s/epoch - 4ms/step
    Epoch 10/20
    Restoring model weights from the end of the best epoch: 4.
    2260/2260 - 8s - loss: 0.8157 - val_loss: 1.1805 - 8s/epoch - 4ms/step
    Epoch 10: early stopping
    


```python
plt.plot(r3.history['loss'], label = 'train loss')
plt.plot(r3.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()
```


    
![png](\images\portfolio\RecommendationSystem\output_104_0.png)
    



```python
hybrid_model =  tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/Projects/Recommender System/model(v2)')
```


```python
from sklearn.metrics import mean_squared_error
y_pred = hybrid_model.predict([test_user, test_product, test_feature_emb])
y_true = test_ratings

#  Compute RMSE
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))
```

    
    
    Testing Result With Keras Matrix-Factorization: 1.0528 RMSE
    


```python
1.062-1.0528
```




    0.009200000000000097



As we can see by using a hybrid model we actually manage to further improve the performance of our recommendation system by about 0.0092 RMSE which isn't that much of an improvement on fewer epoch. But we have to remember that the benefit of using a hybrid model can't only be measured through the decrease in loss, we have to consider it's ability to combat the cold-start problem and recommend unpopular items.

# Get Recommendations

For our recommendation we will use our model to predict the ratings for products that a user has never bought, then we will take 10 product with the highest predicted ratings and recommend them to our user.


```python
# Select a random user from the dataset
user_id = merge_df.UserID.sample(1).iloc[0]

# Products bought by the customer
product_bought = merge_df[merge_df.UserID == user_id]

# Products the user never bought
product_not_bought = np.array(list(set(merge_df[merge_df.UserID != user_id]["NewProductID"])))
feature_product_not_bought = np.array(list(merge_df.drop_duplicates(subset="ProductID")['feature_embedding'])) # Word Embeddidngs of these product feature

# Duplicate the user_id so that it has the same length as the number of product not bought
new_user_id =  list(set(merge_df[merge_df["UserID"] == user_id]["NewUserID"]))
new_user_id = np.array(new_user_id * len(product_not_bought))

# Get the predicted ratings of these products the user never bought and select 10 with highest ratingsW
ratings = hybrid_model.predict([new_user_id, product_not_bought, feature_product_not_bought]).flatten()
top_rating_indices = ratings.argsort()[-10:][::-1]
recommended_product_id = list(set(merge_df[merge_df["NewProductID"].isin(top_rating_indices)]["ProductID"]))
```


```python
print("Showing recommendations for user: {}".format(user_id))
print("====" * 20)
print("Product Bought from User with High Ratings")
print("----" * 11)
top_product_bought = (
    product_bought.sort_values(by="Rating", ascending=False)
    .head(5).ProductID.values
)
product_df_rows = clean_meta_df[clean_meta_df["ProductID"].isin(top_product_bought)]
for idx, row in enumerate(product_df_rows.itertuples()):
    print('{}. Title: {} | Brand: {}'.format(idx+1, row.title, row.brand))

print("")
print("Top 10 product recommendations")
print("----" * 11)
recommended_product =  clean_meta_df[clean_meta_df["ProductID"].isin(recommended_product_id)]
for idx, row in enumerate(recommended_product.itertuples()):
    print('{}. Title: {} | Brand: {}'.format(idx+1, row.title, row.brand))

```

    Showing recommendations for user: A3PMLD8SPBPYIF
    ================================================================================
    Product Bought from User with High Ratings
    --------------------------------------------
    1. Title: Flush Mount Rear View Camera - Marine Grade Waterproof 1.25'' Cam Built-in Distance Scale Lines Backup Parking/Reverse Assist IR Night Vision LEDs w/ 420 TVL Resolution &amp; RCA Output - Pyle PLCM22IR | Brand: Pyle
    2. Title:  iPearl mCover Hard Shell Case with FREE keyboard cover for 15&quot; Model A1286 Aluminum Unibody MacBook Pro (Black keys, 15.4-inch diagonal regular display) - BLACK | Brand: mCover
    3. Title: Sanoxy Lightning to USB Sync Data Cable 3.3-Feet/1m for iPhone 5/5s/6/6 Plus  - Retail Packaging - Black | Brand: SANOXY
    4. Title: GR-8 Bluetooth Headphones + BONUS Car Charger &amp; Armband By Hematiter | Up to 8 Hours of Music Best Wireless Earbuds for Sports, Workouts &amp; Running | IPX7 Waterproof Earphones with Premium Sound | Brand: Hematiter
    
    Top 10 product recommendations
    --------------------------------------------
    1. Title: Polk Audio RC60i 2-way Premium In-Ceiling 6.5&quot; Round Speakers, Set of 2 Perfect for Damp and Humid Indoor/Outdoor Placement - Bath, Kitchen, Covered Porches (White, Paintable Grille) | Brand: Polk Audio
    2. Title: Polk Audio RC60i 2-way Premium In-Ceiling 6.5&quot; Round Speakers, Set of 2 Perfect for Damp and Humid Indoor/Outdoor Placement - Bath, Kitchen, Covered Porches (White, Paintable Grille) | Brand: Polk Audio
    3. Title: iSmartOnline Lightning to HDMI Adapter,Lightning Digital AV Adapter to 1080P HD TV for iPhone 8 / 8 Plus, iPhone X, iPhone 7 / 7Plus, iPad Air / Mini/ Pro, iPod, Plug and Play (HDMI adapter) | Brand: NYXCAM
    4. Title:  Canon EF 100mm f/2.8L IS USM Macro Lens for Canon Digital SLR Cameras  | Brand: Canon
    5. Title: Crucial MX500 2TB 3D NAND SATA 2.5 Inch Internal SSD - CT2000MX500SSD1 | Brand: Crucial
    6. Title: Dakota Alert DCMT-2500 Transmitter 2500' (Green) | Brand: Dakota Alert
    7. Title: cdhgtjtyl Rugged Armor A15 1TB 2.5-Inch USB 3.0 Drop Tested MIL-STD-810F Military Grade External Hard Drive, Black (SP010TBPHDA15S3K) | Brand: cdhgtjtyl
    8. Title: Tera Grand - Premium USB 2.0 to RS232 Serial DB9 Adapter - Supports Windows 10, 8, 7, Vista, XP, 2000, 98, Linux and Mac - Built with FTDI Chipset | Brand: Tera Grand
    9. Title: Transcend 128GB SATA III 6Gb/s MTS400 42 mm M.2 SSD Solid State Drive (TS128GMTS400) | Brand: Transcend
    10. Title: Network Adapter, Anker USB 3.0 to RJ45 Gigabit Ethernet Adapter Supporting 10/100/1000 bit Ethernet | Brand: Anker
    11. Title: Teclast 2 in 1 Waterproof Wireless Bluetooth Removable Keyboard + Protective Leather Stand Case Cover for Teclast X98 /X98 Air 3g/p98 3g (Black) | Brand: Teclast
    

## Conclusion

In conclusion, we've taken the Amazon Review Dataset more specifically its Electronic Category and created a collaborative filtering recommendation system using matrix factorizaton, modified matrix factorization with concatentation and dense layer, and a hybrid model. We found out that the hybrid model resulted in the best performance albeit just slightly over the modified collaborative filtering.
