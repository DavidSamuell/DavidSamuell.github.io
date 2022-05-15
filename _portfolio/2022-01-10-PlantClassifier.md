---
title: Dangerous Plant Classifier
date: 2022-01-09
tags:
  - Computer Vision
  - Deep Learning
  - Convolutional Neural Network
author_profile: true
toc: false
classes: wide

---

Recently, I started my deep learning journey on FastAI: *Practical Deep Learning for Coders*. I wanted to create this project to showcase what I've learned on image classification, and so I decided to create this simple dangerous plant classifier and deploy it in the form of a web app using render. You can click the following link to try it yourself [https://dangerous-plants-detector.onrender.com/](https://dangerous-plants-detector.onrender.com/.) 

### Snapshots
![png](\images\portfolio\PlantClassifier\deploy2.png)

![png](\images\portfolio\PlantClassifier\deploy3.png)

## Background
In the past there has been quite a few casualties among childrens who come in contact with poisonous plant in their backyard or garden. With a dangerous plant classifier parents can reduce the risk of unwanted incident by identifying and removing dangerous plants on their backyard. All they have to do is take a picture of plants that seems uncommon or suspicious and upload them to the application.

To create a dangerous plant classifier I'll need to build a multi-label image classifier as it will enable the model to not only classify dangerous plants, but also identify those that are potentially safe. 

## Dangerous Plant Dataset

The model is trained on a custom dataset containing 30 categories of commonly found dangerous plant on a person backyard (The list is based on this [website](https://raisingchildren.net.au/toddlers/safety/poisons/dangerous-plants)). I created the dataset by scraping images from duck duck go search API.


# Code + Explanation


```python
import fastbook
from fastbook import *
from fastai.vision.widgets import *
```

## Image Collection

### Scrape Images of Highly Poisonous Plants


```python
poisonPlants = ['Castor oil plant (Ricinus communis)','Coral tree (Erythrina genus)','Deadly nightshade (Atropa belladonna)', 'Golden dewdrop (Duranta erecta)', 'Toxicodendron succedaneum', 'Chinaberry (fruit)']
path = Path('/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Highly Poisonous')

```


```python
path.mkdir()
for o in poisonPlants:
    dest = (path/o)
    dest.mkdir(exist_ok=True)
    results = search_images_ddg(o)
    download_images(dest, urls=results)
```


```python
fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink);
```


### Scrape Images Dangerous plants to avoid


```python
dangerousPlants = ['Angel’s trumpet (Brugmansia genus)', 'Arum lily (Zantedeschia aethiopica)', 'Amaryllis belladonna', 'Cacti Plant', 'Chillies Plant', 'Daphne', 'Dumb cane (Dieffenbachia genus)', 'water hemlock OR poison hemlock', 'Lantana', 'Mushrooms and toadstools', 'Poinsettia', 'Myrtle spurge', 'Milkweed']
path = Path('/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Dangerous Plants')
```


```python
path.mkdir()
for o in dangerousPlants:
    dest = (path/o)
    dest.mkdir(exist_ok=True)
    results = search_images_ddg(o)
    download_images(dest, urls=results)
```


```python
fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink);
```


### Scrape Images Plants to treat with caution


```python
cautionPlants = ['agapanthus', 'autumn crocus', 'clivia', 'daffodil', 'hippeastrum', 'hyacinth', 'lily of the valley', 'tulips', 'irises', 'grevilleas', 'Parietaria judaica']
path = Path('/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Caution Plants')
```

```python
path.mkdir()
for o in cautionPlants:
    dest = (path/o)
    dest.mkdir(exist_ok=True)
    results = search_images_ddg(o)
    download_images(dest, urls=results)
```

```python
fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink);
```

## Split Training and Test Set

Because it is quite hard to create a test set using the FastAI library I will put the images and labels into a dataframe and split them using Sk-Learn Stratified Train-Test Splits. I decided to use stratified split due to the fact that we have a handful amount of category in our dataset (28), stratified splits make sure that our test set contain plants from every existing category which will help us evaluate our model performance better.

*Without stratified split it is possible for our test set to not contain a plant from a certain category, and we won't be able to find out how well our model is able to classify that certain type of plant).*


```python
import os

FOLDER_PATH = ['Highly Poisonous', 'Dangerous Plants', 'Caution Plants']
ROOT_PATH = '/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/'
for i in FOLDER_PATH:
  sub_path = os.path.join(ROOT_PATH, i)
  sub2_path = Path(sub_path).ls()
  for j in sub2_path:
    print(f"{j.name}: {str(len(j.ls()))}")
```

    Golden Dewdrop (Duranta Erecta): 242
    Deadly Nightshade (Atropa Belladonna): 233
    Chinaberry (Melia azedarach): 234
    Coral Tree (Erythrina): 221
    Toxicodendron succedaneum: 191
    Castor Oil Plant (Ricinus communis): 254
    Angel’s trumpet (Brugmansia genus): 217
    Arum lily (Zantedeschia aethiopica): 248
    Amaryllis belladonna: 242
    Cacti Plant: 270
    Dumb cane (Dieffenbachia genus): 252
    Lantana: 281
    Poinsettia: 274
    Myrtle spurge: 239
    Milkweed: 229
    Mushrooms and Toadstools: 228
    Daphne: 257
    Chilli Plant: 271
    Water Hemlock or Poison Hemlock: 268
    Parietaria judaica: 252
    Tulips: 274
    Lily of the valley: 239
    Irises: 286
    Hyacinth: 267
    Hippeastrum: 271
    Grevilleas: 242
    Daffodil: 279
    Clivia: 254
    Autumn crocus: 263
    Agapanthus: 262
    


```python
fnames = get_image_files(ROOT_PATH)
```


```python
def parent_label_multi(x):
    return [Path(x).parent.name]
```


```python
df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = fnames
df['label'] = df['fname'].apply(parent_label_multi)
```


```python
df.head()
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
      <th>fname</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Highly Poisonous/Golden Dewdrop (Duranta Erecta)/00000007.jpg</td>
      <td>[Golden Dewdrop (Duranta Erecta)]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Highly Poisonous/Golden Dewdrop (Duranta Erecta)/00000000.jpg</td>
      <td>[Golden Dewdrop (Duranta Erecta)]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Highly Poisonous/Golden Dewdrop (Duranta Erecta)/00000002.jpg</td>
      <td>[Golden Dewdrop (Duranta Erecta)]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Highly Poisonous/Golden Dewdrop (Duranta Erecta)/00000006.jpg</td>
      <td>[Golden Dewdrop (Duranta Erecta)]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Highly Poisonous/Golden Dewdrop (Duranta Erecta)/00000001.jpg</td>
      <td>[Golden Dewdrop (Duranta Erecta)]</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

X_train, X_test =  train_test_split(df ,test_size=0.1, random_state=1, stratify = df['label'])
```


```python
X_test
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
      <th>fname</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7234</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Dangerous Plants/Chilli Plant/00000268.jpg</td>
      <td>[Chilli Plant]</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Caution Plants/Parietaria judaica/00000129.JPG</td>
      <td>[Parietaria judaica]</td>
    </tr>
    <tr>
      <th>3611</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Caution Plants/Clivia/00000147.jpg</td>
      <td>[Clivia]</td>
    </tr>
    <tr>
      <th>3423</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Caution Plants/Daffodil/00000235.jpg</td>
      <td>[Daffodil]</td>
    </tr>
    <tr>
      <th>3355</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Caution Plants/Daffodil/00000173.jpg</td>
      <td>[Daffodil]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Dangerous Plants/Arum lily (Zantedeschia aethiopica)/00000042.jpg</td>
      <td>[Arum lily (Zantedeschia aethiopica)]</td>
    </tr>
    <tr>
      <th>4974</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Dangerous Plants/Cacti Plant/00000028.jpg</td>
      <td>[Cacti Plant]</td>
    </tr>
    <tr>
      <th>6021</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Dangerous Plants/Myrtle spurge/00000003.jpg</td>
      <td>[Myrtle spurge]</td>
    </tr>
    <tr>
      <th>1332</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Highly Poisonous/Castor Oil Plant (Ricinus communis)/00000242.jpg</td>
      <td>[Castor Oil Plant (Ricinus communis)]</td>
    </tr>
    <tr>
      <th>6740</th>
      <td>/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/Dangerous Plants/Daphne/00000026.jpg</td>
      <td>[Daphne]</td>
    </tr>
  </tbody>
</table>
<p>751 rows × 2 columns</p>
</div>



Great! Now we've successfully split the data into training and test set.

## Load Data


Here we set up our datablock, which will take the various plant images we've collected and prepare them into minibatches which contain our training and validation set.


```python
path = Path('/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification')
```


```python
def get_x(r): return r['fname']
def get_y(r): return r['label']
```

### Data Augmentation

Before we load our data in mini-batches, we will apply some augmentation to our data. First, we will resize every individual image in the dataset to a size of 460x460 pixels so that they can be processed together as a batch. Batch transformation can be then apply, instead of applying the transformation to individual image it can take advantage of the GPU and simultaneously transform all the item in a mini-batch(in our case 64 images per batch). We will use FastAI augmentive transform to apply the batch transformation.

What the transformation does is basically pick a random scaled crop of the image, perform various transformation like warping, rotating, zooming in, brightness change, and contrast change, and finally resize it to a final size of 224 pixels. This will allow our model to better identify different features of the same object.


```python
plants_multi = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock), 
    get_x = get_x, 
    get_y = get_y,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224, min_scale=0.75)
)
# The batch size is set to default which is 64.

dls = plants_multi.dataloaders(X_train)
dsets = plants_multi.datasets(X_train)

```


```python
dsets.train[0]
```




    (PILImage mode=RGB size=1698x1131,
     TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))




```python
print(dsets.train.vocab)

```

    ['Agapanthus', 'Amaryllis belladonna', 'Angel’s trumpet (Brugmansia genus)', 'Arum lily (Zantedeschia aethiopica)', 'Autumn crocus', 'Cacti Plant', 'Castor Oil Plant (Ricinus communis)', 'Chilli Plant', 'Chinaberry (Melia azedarach)', 'Clivia', 'Coral Tree (Erythrina)', 'Daffodil', 'Daphne', 'Deadly Nightshade (Atropa Belladonna)', 'Dumb cane (Dieffenbachia genus)', 'Golden Dewdrop (Duranta Erecta)', 'Grevilleas', 'Hippeastrum', 'Hyacinth', 'Irises', 'Lantana', 'Lily of the valley', 'Milkweed', 'Mushrooms and Toadstools', 'Myrtle spurge', 'Parietaria judaica', 'Poinsettia', 'Toxicodendron succedaneum', 'Tulips', 'Water Hemlock or Poison Hemlock']
    


```python
x,y = dls.one_batch()
```


```python
x.shape
```




    torch.Size([64, 3, 224, 224])




```python
y.shape
```




    torch.Size([64, 30])




```python
dls.show_batch(max_n=9, figsize=(7,8))
```


    
![png](\images\portfolio\PlantClassifier\output_37_0.png)
    


Here we can check the various augmentation applied by fastAI augmentive transform on a single image.. 


```python
dls.show_batch(max_n=9, figsize=(7,8), unique=True)
```


    
![png](\images\portfolio\PlantClassifier\output_39_0.png)
    


### Load Test Data

Now let's set up the datablock for our test set, the process is a little bit complicated to perform in fastAI, but fortunately I found this helpful [article](https://muellerzr.github.io/fastblog/2020/08/10/testdl.html) from Zachary Mueller.



```python
p = Pipeline([ColReader('fname'), PILImage.create])
```


```python
dls.valid_ds.tls[0].tfms = p
```


```python
dls.valid_ds.tls[0].types.insert(0, pd.Series)
```


```python
dls.valid_ds.tls[0].types[1:]
```




    [pandas.core.series.Series,
     (pathlib.Path, str, torch.Tensor, numpy.ndarray, bytes),
     fastai.vision.core.PILImage]




```python
test_dl = dls.test_dl(X_test, with_labels=True)
```


```python
test_dl.vocab
```




    ['Agapanthus', 'Amaryllis belladonna', 'Angel’s trumpet (Brugmansia genus)', 'Arum lily (Zantedeschia aethiopica)', 'Autumn crocus', 'Cacti Plant', 'Castor Oil Plant (Ricinus communis)', 'Chilli Plant', 'Chinaberry (Melia azedarach)', 'Clivia', 'Coral Tree (Erythrina)', 'Daffodil', 'Daphne', 'Deadly Nightshade (Atropa Belladonna)', 'Dumb cane (Dieffenbachia genus)', 'Golden Dewdrop (Duranta Erecta)', 'Grevilleas', 'Hippeastrum', 'Hyacinth', 'Irises', 'Lantana', 'Lily of the valley', 'Milkweed', 'Mushrooms and Toadstools', 'Myrtle spurge', 'Parietaria judaica', 'Poinsettia', 'Toxicodendron succedaneum', 'Tulips', 'Water Hemlock or Poison Hemlock']




```python
test_dl.show_batch(max_n=9, figsize=(7,8))
```


    
![png](\images\portfolio\PlantClassifier\output_48_0.png)
    


## Model

For our model we'll use a **ResNet-18** that has been pre-trained on the ImageNet dataset. We can just call the `cnn_learner` from fastAI and pass it in our data loader, architecture, and metric. The metric we will use is accuracy multi. It will select activations that are higher than the determined threshold (0.5) as true or 1 and lower as false or 0. 

For our loss function we don't have to pass it in to our learner as fastAI will chose the appropriate one.


```python
learn = cnn_learner(dls, resnet18,  metrics=partial(accuracy_multi, thresh=0.5), model_dir = "/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/models").to_fp16()
```


```python
learn.loss_func
```




    FlattenedLoss of BCEWithLogitsLoss()



As we can see FastAI automatically assigned the appropriate loss function according to our DataBlock. Since this is a multilabel classification it assign the BCEWithLogitsLoss loss function which is basically just a binary cross entropy with sigmoid applied to the activations.

Before we start training we can use a [learning rate finder](https://arxiv.org/abs/1506.01186) to find the "*perfect*" learning rate for our model. This brilliant idea came up from a researcher named Leslie Smith, what he basically propose is we keep track of the change in loss when trainig one mini batches using different learning rates. We start with a very small learning rates and gradually increase it by some percentage, we keep doing this untill the loss gets worse, instead of better.

Using the `lr_find` function from fastAI it will give us a plot of these loss with different learning rates. Then as a rule of thumb we can select a learning rate where our loss is decreasing steeply or one order of magnitude less than where the minimum loss was achieved (i.e., the minimum divided by 10). Based on the graph below we'll select a value of 5e-2,


```python
lr_min, lr_steep, lr_valley = learn.lr_find(suggest_funcs=(minimum, steep, valley))
```


    
![png](\images\portfolio\PlantClassifier\output_55_2.png)
    



```python
print(f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}")
```

    Minimum/10:	6.31e-02
    Steepest point:	3.98e-02
    Longest valley:	4.37e-03
    

Since we're using a pre-trained model we will be performing transfer learning. Where first we will fine tune the final layers or head of our model for a few epochs. Then we unfreeze the rest of the layers, and finally we train all of the layers for even longer epochs

For now we will start training the final layers for 5 epochs with the learning rate we've chosen, we will use the [one cycle policy](https://arxiv.org/abs/1708.07120) proposed by the same person who proposed the learning rate finder: Leslie Smith. The general idea is a schedule for learning rate which is separated into two phases: one where the learning rate grows from the minimum value to the maximum value (warmup), and one where it decreases back to the minimum value (annealing). Doing so will allows us to train with a much higher maximum learning rate which will bring 2 benefit: 
1.  Faster training speed.
2.  Less overfitting due to skipping sharp local minima. 

We can use the one cycle policy in fastai by calling the `fit_one_cycle` function.


```python
learn.fit_one_cycle(5, lr_max=5e-2, cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3)])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.252400</td>
      <td>0.138229</td>
      <td>0.966642</td>
      <td>05:41</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.109182</td>
      <td>0.076304</td>
      <td>0.976684</td>
      <td>05:46</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.071792</td>
      <td>0.056568</td>
      <td>0.981397</td>
      <td>05:42</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.055242</td>
      <td>0.042013</td>
      <td>0.986553</td>
      <td>05:46</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.042078</td>
      <td>0.038352</td>
      <td>0.987762</td>
      <td>05:47</td>
    </tr>
  </tbody>
</table>
    


```python
learn.save('stage-1')
# learn.load('stage-1-v2')
```




    Path('/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/models/stage-1.pth')



Unfreeze the body of our model


```python
learn.unfreeze()
```

Use another learning rate finder to find the best learning rate for training our whole model.


```python
lr_min, lr_steep, lr_valley = learn.lr_find(suggest_funcs=(minimum, steep, valley))
```





    /usr/local/lib/python3.7/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 12 bytes but only got 10. Skipping tag 42037
      " Skipping tag %s" % (size, len(data), tag)
    


    
![png](\images\portfolio\PlantClassifier\output_63_2.png)
    



```python
print(f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}")
```

    Minimum/10:	3.31e-05
    Steepest point:	1.58e-06
    Longest valley:	4.37e-05
    

For the training process after unfreezing, we will use the discriminative learning rate proposed by [Jason Yosinski](https://arxiv.org/abs/1411.1792). The technique suggest that different layers should be train at different speeds, earlier layers should be slower since they study general task like edge and gradient which are transferable to our task, while later layers which are more specific to the pre-train task should be train faster so that it fits with the specific task we are trying to solve.

To use a discriminative learning rate we can pass in the slice object to the `lr_max` parameter. The first parameter in the slice object will be the learning rate use in the earliest layer of our neural network, and the second value will be the learning rate in the final layer. The layers in between will have learning rates that are multiplicatively equidistant throughout that range. 


```python
learn.fit_one_cycle(12, lr_max=slice(2e-6, 3e-4), cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3), SaveModelCallback(monitor='accuracy_multi', fname='best-pre-trained-restnet')])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.038314</td>
      <td>0.038239</td>
      <td>0.987491</td>
      <td>05:52</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.037085</td>
      <td>0.036238</td>
      <td>0.987886</td>
      <td>05:47</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.034705</td>
      <td>0.034821</td>
      <td>0.988724</td>
      <td>05:53</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.033563</td>
      <td>0.033108</td>
      <td>0.988675</td>
      <td>05:55</td>
    </tr>
  </tbody>
</table>
    

Not bad by training using a simple pre-trained model we managed to achieve a close to 99% accuracy on our validation set. Now, let us try to evaluate our model on the test set.


```python
# learn.export()
# learn.save('best-pre-trained-restnet')
```




    Path('/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/models/best-pre-trained-restnet.pth')




```python
learn.show_results(figsize=(15,15))
```






    
![png](\images\portfolio\PlantClassifier\output_69_1.png)
    


## Evaluation
To evaluate our model performance we can test it on the test set and check its loss and accuracy.


```python
learn.load('best-pre-trained-restnet')
```




    <fastai.learner.Learner at 0x7f3f3b547110>




```python
learn.validate(dl=test_dl)
```








    (#2) [0.08884591609239578,0.9748779535293579]




```python
preds, targs = learn.get_preds(dl= test_dl)
accuracy_multi(preds, targs, sigmoid=False, thresh = 0.5)
```








    TensorBase(0.9749)



As we can see we have achieve a satisfying result with only 97.5% accuracy. This shows that our model are able to perform well even on data it hasn't seen before.

### Classification Report

Next up we can print out a classification report which will show us the precision, recall, and f1-score of each class. We'll use the help of sklearn `classification_report` to do so. But first we'll need to convert the prediction result from our model so that it is compatible with sklearn classification report function.




```python
preds2 = preds.detach().clone()
```


```python
def preds_converter(inp, thresh=0.5):
  for i, x in enumerate(inp):
    for j, y in enumerate(x):
      if(y > thresh):    
        preds2[i][j] = 1
      else:
        preds2[i][j] = 0
```


```python
preds_converter(preds)
```

We can use the precision and recall result to identify which class our model is having trouble predicting. To demonstrate how I use this information to improve my model, let's take a look at the accuracy and classficiation report of my initial experiments:

![png](\images\portfolio\PlantClassifier\Accuracy.png)
![png](\images\portfolio\PlantClassifier\ClassificationReport.png)


As you can see intially my model achieve an accuracy of 97. From the classification report we can see three classes with a very low recall: `euphorbia genus` , `Rhus or wax tree`, and `White cedar tree`. This shows that a large number of the images in these classes are undetected by the model. After some further investigation I found out that the images in those classes varies from one another.

For example Euphorbia Genus is actually a very large and diverse genus of flowering plants, if we look it up in our browser we'll find various plants  that aren't even remotely similar. So to fix these issue a simple keyword fix should solve it. For example, instead of searching for `euphorbia genus` we should search for the subspecies that are poisonous:  Poinsettia, Myrtle spurge, Milkweed.

So I try to fix this issue by replacing the appropriate keywords for image scraping and retrain our model. And just like that the accuracy improve from 97% to 97.5% which might not seem that much, but still an improvement nonetheless.

Here is the classification of our current model after we've fixed the dataset:


```python
from sklearn.metrics import classification_report
print(classification_report(targs, preds2, target_names = dls.vocab))
```

                                           precision    recall  f1-score   support
    
                     Amaryllis belladonna       0.92      0.90      0.91        40
       Angel’s trumpet (Brugmansia genus)       1.00      0.74      0.85        43
      Arum lily (Zantedeschia aethiopica)       0.95      0.80      0.87        49
                              Cacti Plant       0.91      0.67      0.77        64
      Castor oil plant (Ricinus communis)       0.89      0.56      0.68        45
                           Chillies Plant       0.83      0.75      0.78        51
                       Chinaberry (fruit)       0.84      0.57      0.68        47
             Coral tree (Erythrina genus)       0.89      0.45      0.60        38
    Deadly nightshade (Atropa belladonna)       0.97      0.78      0.86        40
          Dumb cane (Dieffenbachia genus)       0.80      0.82      0.81        44
          Golden dewdrop (Duranta erecta)       0.87      0.74      0.80        46
                                  Lantana       1.00      0.88      0.93        48
                                 Milkweed       0.85      0.66      0.74        50
                 Mushrooms and toadstools       0.95      0.78      0.86        46
                            Myrtle spurge       0.86      0.76      0.81        41
                       Parietaria judaica       0.89      0.80      0.84        50
                               Poinsettia       0.97      0.88      0.93        43
                Toxicodendron succedaneum       0.89      0.25      0.39        32
                               agapanthus       0.96      0.96      0.96        47
                            autumn crocus       0.92      0.86      0.89        42
                                   clivia       0.91      0.93      0.92        44
                                 daffodil       0.90      0.88      0.89        32
                            daphnes plant       0.94      0.65      0.77        46
                               grevilleas       0.94      0.71      0.81        41
                              hippeastrum       0.95      0.83      0.88        46
                                 hyacinth       0.95      0.86      0.90        43
                                   irises       0.98      0.83      0.90        52
                       lily of the valley       0.93      0.83      0.87        46
                                   tulips       0.91      0.87      0.89        46
          water hemlock OR poison hemlock       0.88      0.71      0.79        49
    
                                micro avg       0.92      0.76      0.83      1351
                                macro avg       0.91      0.76      0.82      1351
                             weighted avg       0.91      0.76      0.82      1351
                              samples avg       0.75      0.76      0.75      1351

There are still some classes with low recall score like `Toxicodendron Succedaneum`, `Castor Oil Plant`, and `Coral Tree`. This shows that there is still room for further improvement on our dataset. 


## Inference
We know that our model is quite accurate at predicting the plants that are poisonous or dangerous as shown in the accuracy of our test set, but will it be able to distinguish plants that aren't dangerous or poisonous at all?


```python
def get_x(r): return r['fname']
def get_y(r): return r['label']
```


```python
learn = load_learner('/content/drive/MyDrive/Colab Notebooks/FastAI - Notebook/Mini-Projects/Plants Classification/models/best-pre-trained-restnet.pkl')
```


```python
btn_upload = widgets.FileUpload()
btn_run = widgets.Button(description='Classify')
out_pl = widgets.Output()
lbl_pred = widgets.Label()
```


```python
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn.predict(img)
    if(len(pred)==0):
      lbl_pred.value = "Plant is potentially safe!"
    else:
      lbl_pred.value = f'Prediction: {pred} Probability: {probs[pred_idx]}'
    lbl_pred

btn_run.on_click(on_click_classify)
```


```python
#hide_output
VBox([widgets.Label('Select your plant'), 
      btn_upload, btn_run, out_pl, lbl_pred])
```


![png](\images\portfolio\PlantClassifier\Inference.png)





As we can see our model can recognize that a banana tree doesn't belong to any of the poisonous/dangerous plant species it is trained with.

## Future Notes
Overall I am quite satisfied with the result of this experiments, some sugestions that I would give to further improve it in the future would probably be getting a hand on experts opinion regarding the datasets. They could help us ensure the quality of the datasets by making sure that the images we have are representative of every possible scenarios (some plants may have different colours, or grow flowers and fruits during specific seasons).
