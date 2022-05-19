---
title: License Plate Super Resolution
excerpt: "<img src='/images/portfolio/SuperResolution(2).png'>
<br><br>
On my 5th semester in Binus University I took a Computer Vision course and for the final project me and my 2 teammates chose to build a super-resolution model (GAN) to enhance blurry unreadable linense plate into readable ones (License Plate Image Enhancer). Although this topic is quite advanced for the course at the time I found it interesting and so we decide to pursue it.
<br><br>
<b>Tags: Computer Vision | Deep Learning | GAN | FastAI (PyTorch)</b>"
date: 2022-05-18
tags:
  - Deep Learning
  - GAN
  - Super-Resolution
  - Data Science
collection: portfolio
---


On my 5th semester in Binus University I took a Computer Vision course and for the final project me and my 2 teammates chose to build a super-resolution model (GAN) to enhance blurry unreadable linense plate into readable ones (License Plate Image Enhancer). Although this topic is quite advanced for the course at the time I found it interesting and so we decide to pursue it.

Thanks to Jeremy Howard FastAI course I manage to gathered the necessary knowledge and tools to build a super resolution model using GAN. I learnt a lot of new stuff such as U-Net and ResNet architecture, skip connections, and general adversarial loss. Well enough of the introduction and let's get into coding.

# Dataset

For training, we use data from [The Chinese City Parking Dataset](https://github.com/detectRecog/CCPD/), which contains more than 200k images of cars under numerous conditions. The dataset also provides annotations that allow us to crop out plates from images as the raw data cannot be consumed by our model, which only requires the image of license plates without the street views and cars. Meanwhile, for testing, we use data from the [Indonesian License Plate Dataset](https://www.kaggle.com/imamdigmi/indonesian-plate-number), which consists of roughly 500 images of Indonesian license plates.

# Preprocess the Data

We preprocess the CCPD (The Chinese City Parking Dataset) by cropping out plates from the images by utilizing the annotations in the dataset since as mentioned previously, we do not want the street views and cars in the images and only want the image of the license plates. Furthermore, we want to remove images with poor brightness and contrast and preserve just the good ones as labels. 
An example of what the image look like before and after preprocessing is shown here:

![png](/images/portfolio/LicensePlateSuperResolution/snippet_1.png)

To perform the pre-processing you can simply follow the direction in the  [The Chinese City Parking Dataset](https://github.com/detectRecog/CCPD/) README file. 

On a sidenote I didn't use the whole 200k images for training as it will simply take a lot of time and I don't have the necessary computing power, so instead I sample around 3000 images and save it into a Mini-Dataset folder. This "Mini-Dataset" is what we'll use to train our model (generator and critic).

# Data Crappifier

Now that we have a dataset full of license plates, we want to artificially create an input and label pair where the inputs are the LR (Low Resolution) version of the license plate images from the CCPD which are obtained by downsampling them, whereas the labels are the original images.

We can create the crappy lower version of our license plate dataset using the following code:


```python
#hide
!pip install -Uqq fastbook
import fastbook
# fastbook.setup_book()
```

    [K     |████████████████████████████████| 720 kB 12.1 MB/s 
    [K     |████████████████████████████████| 46 kB 4.1 MB/s 
    [K     |████████████████████████████████| 189 kB 55.0 MB/s 
    [K     |████████████████████████████████| 1.2 MB 45.5 MB/s 
    [K     |████████████████████████████████| 56 kB 4.7 MB/s 
    [K     |████████████████████████████████| 51 kB 351 kB/s 
    [?25h


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
from fastbook import *
```


```python
class Crappifier():
  "Quickly draw tesxt and numbers on an image"
  def __init__(self, path_lr, path_hr):
      self.path_lr = path_lr
      self.path_hr = path_hr              
      
  def __call__(self, fn):       
      dest = self.path_lr/fn.relative_to(self.path_hr)    
      dest.parent.mkdir(parents=True, exist_ok=True)
      img = Image.open(fn)
      q = random.randint(1, 3)
      img.save(dest, quality=q)
```


```python
#
path = Path('/content/drive/MyDrive/Project Comvis')
```


```python
path.ls()
```




    (#1) [Path('/content/drive/MyDrive/Project Comvis/Mini-Dataset')]




```python
#
path_hr = path/'Mini-Dataset'
path_lr = path/'Crappy-Dataset'
```


```python
items = get_image_files(path_hr)
```


```python
parallel(Crappifier(path_lr, path_hr), items);
```


```python
bad_im = get_image_files(path_lr)
```


```python
im1 = PILImage.create(items[0])
im2 = PILImage.create(bad_im[0])
```


```python
im1.show(); im2.show(figsize=(5,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd828202710>




    
![png](\images\portfolio\LicensePlateSuperResolution\output_19_1.png)
    



    
![png](\images\portfolio\LicensePlateSuperResolution\output_19_2.png)
    


The following figure shows the comparison between our license plate after it has gone through down sampling:

![png](/images/portfolio/LicensePlateSuperResolution/snippet_2.png)

# Data Augmentation

Afterwards, we create a datablock which acts as a blueprint of how our data will be loaded into the models. In our DataBlock We specify the split of our dataset (CCPD) to be 80% training and 20% validation. We also specify some transformations or augmentations to be applied to our data for preprocessing purposes which include brightness and contrast change to simulate the license plates’ conditions when exposed to sunlight, or conversely, in a dark environment. We don't perform any rotation or warping since we already have a large enough dataset with car license plates under various possible angle.

After the datablock is created, we want to create our dataloaders by passing our images (LR and HR images) to the datablock so the split and transformations can be applied to the images. 

To clarify things Datablock is basically a blueprint on how to assemble the data, it's a special object that comes from the fastAI library. When we pass a source to our datablock we can then convert it to a dataloaders, which we will use to load our data to our model. As we can see we use a batch size of 32, so during each epoch our model will process a single batch at every iteration which contain 32 images.


```python
# 0.4 and 0.7 is the probability of the brighteness
tfms = [Brightness(max_lighting=0.4, p =0.7), Contrast(max_lighting=0.4, p=0.7)]
```


```python
#
dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=get_image_files,
                   get_y = lambda x: path_hr/x.name,
                   splitter=RandomSplitter(valid_pct=0.2, seed=42),
                   batch_tfms=tfms
                   )
dls = dblock.dataloaders(path_lr, bs=32, path = path)
dls.c = 3
```


```python
dls.show_batch(max_n=6, nrows=2)
```


    
![png](\images\portfolio\LicensePlateSuperResolution\output_25_0.png)
    


# Pre-Trained Generator

Now let's create our generator, our generator will be the one to generate our supposedly high quality image from a lower quality one. We'll be using a U-Net learner with a ResNet-34 encoder that is pretrained on ImageNet. 

We also apply some hyperparameters like a weight decay of 10-3 to reduce chance of overfitting, and self attention. For this part, we use Pixel MSE as the loss function. 



```python
wd = 1e-3
y_range = (-3.,3.)
loss_gen = MSELossFlat()
arch = models.resnet34
```


```python
def create_gen_learner():
    return unet_learner(dls, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)
```


```python
learn_gen = create_gen_learner()
```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
    


      0%|          | 0.00/83.3M [00:00<?, ?B/s]


Next we want to train the final layers of the generator for 10 epochs before we unfreeze the other layers and train it for another 3 epochs with a discriminative learning rate ranging from 10^-6 to 10^-4. We picked this learning rate as evident by our `Learning Rate Finder`.

For a bit of background `Learning Rate Finder` is a concept implemented by the FastAI library which enables us to find the "optimal learning rate”. This idea is proposed by researcher Leslie Smith in his paper “Cyclical Learning Rates for Training Neural Networks”. What he proposes is to keep track of the change in loss when training one mini batch using different learning rates. We will start with a very small learning rate and gradually increase it by some percentage until the loss gets worse, instead of better.


Once it's done training we can save the results (generated images) so we can pre-train the critic with it.  

Originally a generator is train directly through the min-max game of the GAN, but this approach can take a really long time before the generator can generate good images. By using a pre-trained generator and critic, we can essentially speed up the training process of the GAN and receieve better generated images in less time.


```python
learn_gen.fit_one_cycle(10, pct_start=0.8, cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3)])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.263082</td>
      <td>0.174527</td>
      <td>07:37</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.168479</td>
      <td>0.143965</td>
      <td>07:36</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.144982</td>
      <td>0.128599</td>
      <td>07:36</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.132131</td>
      <td>0.118273</td>
      <td>07:36</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.124950</td>
      <td>0.112624</td>
      <td>07:36</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.119412</td>
      <td>0.108192</td>
      <td>07:34</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.112668</td>
      <td>0.103837</td>
      <td>07:34</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.108194</td>
      <td>0.098086</td>
      <td>07:33</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.102833</td>
      <td>0.097945</td>
      <td>07:33</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.095219</td>
      <td>0.091305</td>
      <td>07:32</td>
    </tr>
  </tbody>
</table>



```python
learn_gen.unfreeze()
```


```python
# Learning Rate Finder
lr_min, lr_steep, lr_valley = learn_gen.lr_find(suggest_funcs=(minimum, steep, valley))
```






    
![png](\images\portfolio\LicensePlateSuperResolution\output_34_1.png)
    



```python
print(f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}")
```

    Minimum/10:	1.20e-06
    Steepest point:	1.32e-06
    Longest valley:	4.79e-06
    


```python
# Intially we train for 5 epoch but because there is no improvement in the validation loss after 2 epoch early stopping is initiated.
learn_gen.fit_one_cycle(5, slice(1e-6,1e-4), cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=2)])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.093898</td>
      <td>0.091138</td>
      <td>07:56</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.094041</td>
      <td>0.090799</td>
      <td>07:56</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.092767</td>
      <td>0.090555</td>
      <td>07:56</td>
    </tr>
  </tbody>
</table>


    No improvement since epoch 0: early stopping
    


```python
learn_gen.save('gen-pre2')
```




    Path('/content/drive/MyDrive/Project Comvis/models/gen-pre2.pth')




```python
learn_gen.show_results(max_n=4, figsize=(12,12))
```






    
![png](\images\portfolio\LicensePlateSuperResolution\output_38_1.png)
    


## Save the generated images

Now we need these generated images saved away so we can use them to pre-train our critic.


```python
learn_gen.load('gen-pre2')
```




    <fastai.learner.Learner at 0x7f4001b308d0>




```python
#
name_gen = 'Generated Image'
path_gen = path/name_gen
```


```python
path_gen.mkdir(exist_ok=True)
```


```python
def save_preds(dls, learn):
  "Save away predictions"
  names = dls.dataset.items
  
  preds,_ = learn.get_preds(ds_idx = 0)
  for i,pred in enumerate(preds):
      dec = dls.after_batch.decode((TensorImage(pred[None]),))[0][0]
      arr = dec.numpy().transpose(1,2,0).astype(np.uint8)
      Image.fromarray(arr).save(path_gen/names[i].name)
```


```python
save_preds(dls, learn_gen)
```






```python
path_g = get_image_files(path/name_gen)
path_i = get_image_files(path/'Mini-Dataset')
fnames = path_g + path_i
```


```python
fnames[1]
```




    Path('/content/drive/MyDrive/Project Comvis/Generated Image/00979765325671-90_90-271&521_460&581-459&576_268&582_270&520_461&514-0_0_16_16_27_31_29-138-17.jpg')



# Training the critic

Next, to train our critic we will need to create a dataloader by passing our generated image and high-resolution image from our training set. After that, we can create our critic learner by passing our dataloaders to it. 


```python
# Create our critic data loader.
def get_crit_dls(fnames, bs:int):
  #"Generate two `Critic` DataLoaders"
  splits = RandomSplitter(0.1, seed = 42)(fnames)
  dsrc = Datasets(fnames, tfms=[[PILImage.create], [parent_label, Categorize]],
                splits=splits)
  tfms = [ToTensor()]
  gpu_tfms = [IntToFloatTensor()]
  return dsrc.dataloaders(bs=bs, after_item=tfms, after_batch=gpu_tfms, path = path)
```


```python
dls_crit = get_crit_dls(fnames, bs=32)
```


```python
dls_crit.show_batch()
```


    
![png](\images\portfolio\LicensePlateSuperResolution\output_52_0.png)
    


## Critic Learner

Now we're ready to create our learner or model.

We'll use a custom architecture from FastAI for our critic, it is a ResNet based architecture with self attention and spectral normalization. We won't go into much detail here. To get the architecture we can just call `gan_critic`.

As for the loss function, we use a Binary Cross Entropy with Adaptive Loss. We then train our critic for 10 epochs with a learning rate of 5×10-5, again we used a learning rate finder to come up with that learning rate. During training, our model performed early stopping at 3 epochs as there are no noticeable improvements in performance.


```python
from fastai.vision.all import *
from faastai.vision.gan import *
```


```python
loss_crit = AdaptiveLoss(nn.BCEWithLogitsLoss())
```


```python
def create_crit_learner(dls, metrics):
  return Learner(dls, gan_critic(), metrics=metrics, loss_func=loss_crit)
```


```python
learn_crit = create_crit_learner(dls_crit, accuracy_thresh_expand)
```


```python
lr_min, lr_steep, lr_valley = learn_crit.lr_find(suggest_funcs=(minimum, steep, valley))
```






    
![png](\images\portfolio\LicensePlateSuperResolution\output_59_1.png)
    



```python
print(f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}")
```

    Minimum/10:	5.75e-05
    Steepest point:	7.59e-05
    Longest valley:	6.31e-05
    


```python
learn_crit.fit_one_cycle(10, lr_max = 5e-5, wd=wd, cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=2)])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_thresh_expand</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.687624</td>
      <td>0.685341</td>
      <td>0.560556</td>
      <td>14:37</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.686513</td>
      <td>0.690043</td>
      <td>0.529044</td>
      <td>14:36</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.681447</td>
      <td>0.685759</td>
      <td>0.541556</td>
      <td>14:36</td>
    </tr>
  </tbody>
</table>


    No improvement since epoch 0: early stopping
    


```python
# Save our trained critic
learn_crit.save('critic-pre2')
```




    Path('/content/drive/MyDrive/Project Comvis/models/critic-pre2.pth')



# The GAN

Now that we have trained generator and critic, we can create our GAN learner which will train our model in an adversarial fashion — generator vs discriminator in a min-max game. 


```python
dls_crit = get_crit_dls(fnames, bs=32)
```


```python
# Load our trained critic
learn_crit = create_crit_learner(dls_crit, metrics=None).load('critic-pre2')
```


```python
# Load our trained generator
learn_gen = create_gen_learner().load('gen-pre2')
```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
    


      0%|          | 0.00/83.3M [00:00<?, ?B/s]



```python
class GANDiscriminativeLR(Callback):
    "`Callback` that handles multiplying the learning rate by `mult_lr` for the critic."
    def __init__(self, mult_lr=5.): self.mult_lr = mult_lr

    def begin_batch(self):
        "Multiply the current lr if necessary."
        if not self.learn.gan_trainer.gen_mode and self.training: 
            self.learn.opt.set_hyper('lr', learn.opt.hypers[0]['lr']*self.mult_lr)

    def after_batch(self):
        "Put the LR back to its value if necessary."
        if not self.learn.gan_trainer.gen_mode: self.learn.opt.set_hyper('lr', learn.opt.hypers[0]['lr']/self.mult_lr)
```


```python
switcher = AdaptiveGANSwitcher(critic_thresh=.65)
```

To create our GAN learner, we pass in our generator and critic that we've previously trained. 

As for the loss function, we use the weighted sum of the Pixel MSE loss and Critic loss. The reason we don't want to only use the Critic loss as our GAN loss function is because we don't want our GAN to generate picture that look like high resolution image but has nothing to do with our real license plate.

Since the Pixel MSE loss and Critic loss are on a different scale, we multiply the pixel loss by around 50 (1:50 ratio). Another thing about GAN is that there is no reason to add momentum during training because it keep switching between critic and generator, that is why we set an AdamOptimizer with 0 momentum.

We then pick a reasonably low learning rate of 10^-4 to ensure that the model doesn't diverge and a weight decay of 10^-3 to prevent it from overfitting.


```python
# Pass in our generator and critic to the GAN learner
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=True, switcher=switcher,
                                 opt_func=partial(Adam, mom=0.), cbs=GANDiscriminativeLR(mult_lr=5.), path = path)
```


```python
lr = 1e-4
wd = 1e-3
```


```python
#
learn.load('gan-1c')
```




    <fastai.vision.gan.GANLearner at 0x7f70f17d8750>



One of the thing about GAN is that these generator and critic loss are meaningless, you can't expect them to go down because as the generator gets better (gen_loss decrease) it gets harder for the critic to differentiate the real and generated image (critic_loss increase), and then when the critic improve (critic_loss decrease) it gets harder for the generator to create image that can fool the critic (gen_loss increase).

So one way to know how they are performing is to take a look at the result yourself, you can stop the training once you're satisfied with the quality of the image that your GAN is able to generate.

As you can see below it seems like I only train it for 1 epoch, but what happen is I train it for 5-8 epoch multiple times untill I am satisfied with the result. Unfortunately, I lost track of how many epochs I've trained the model but I think it's around 30-40. Another reason I didn't just train it in one go for 10 epochs or more is due to the limited resource of Google Collab GPU.


```python
#learn.fit(1, lr, wd=wd)
```

    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (generator) that exists in the learner. Use `self.learn.generator` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (critic) that exists in the learner. Use `self.learn.critic` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (gen_mode) that exists in the learner. Use `self.learn.gen_mode` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>gen_loss</th>
      <th>crit_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.777248</td>
      <td>5.488672</td>
      <td>5.488672</td>
      <td>0.686626</td>
      <td>1:20:51</td>
    </tr>
  </tbody>
</table>



```python
# Save the currently trained GAN so that I can re-train it when the GPU resources are available
learn.save('gan-1c')
```




    Path('/content/drive/MyDrive/Project Comvis/models/gan-1c.pth')




```python
learn.show_results(max_n=10)
```

    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (generator) that exists in the learner. Use `self.learn.generator` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (critic) that exists in the learner. Use `self.learn.critic` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (gen_mode) that exists in the learner. Use `self.learn.gen_mode` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    






    
![png](\images\portfolio\LicensePlateSuperResolution\output_77_2.png)
    


We can see a comparison of our blurred license plate on the left, our original licence plate on the middle, and our generated super resolution on the right. We can see that it did a pretty good job at generating higher resolution image of our blurred license plate and actually making it readable.

# Inference 

Now that we have our GAN model trained, let's try to generate super resolution images using a new dataset called `Indonesian License Plate`. The dataset contain of over 400 images of Indonesian vehicle license plate. Before we can use it as our data we first need to do the same thing we did previously which is to crappify it.

## Data Crappifier


```python
test_path = path/'Test Set'
```


```python
from PIL import ImageDraw, ImageFont
def resize_to(img, targ_sz, use_min=False):
    w,h = img.size
    min_sz = (min if use_min else max)(w,h)
    ratio = targ_sz/min_sz
    return int(w*ratio),int(h*ratio)
```


```python
class Crappifier():
  "Quickly draw tesxt and numbers on an image"
  def __init__(self, path_lr, path_hr):
      self.path_lr = path_lr
      self.path_hr = path_hr              
      
  def __call__(self, fn):       
      dest = self.path_lr/fn.relative_to(self.path_hr)    
      dest.parent.mkdir(parents=True, exist_ok=True)
      img = Image.open(fn)
      targ_sz = resize_to(img, 96, use_min=True)
      img = img.resize(targ_sz, resample=Image.BILINEAR).convert('RGB')
      q = random.randint(1, 3)
      img.save(dest, quality=q)
```


```python
path_hr = test_path/'Indonesian License Plate'
path_lr = test_path/'Crappy-Test-Dataset'
```


```python
items = get_image_files(path_hr)
```


```python
parallel(Crappifier(path_lr, path_hr), items);
```

## Load our Test Set


```python
crappy_test_path = test_path/'Crappy-Test-Dataset'
save_path = test_path/'Generated-Test-Image'
```


```python
test_files = get_image_files(crappy_test_path)
```


```python
test_files
```




    (#399) [Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/364.E 6528 P-11-18.jpeg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/102.E 6984 P-07-20.jpeg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/173.E 4019 TL-05-22.jpg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/105.E 6430 QO-12-19.jpeg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/119.E 4538 QK-08-19.jpeg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/358.E 3438 SJ-10-19.jpg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/20171211_084300.jpg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/192.E 6347 PAF-09-21.jpg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/111.E 6810 IX-05-20.jpg'),Path('/content/drive/MyDrive/Colab Notebooks/Semester 5 - Computer Vision/Final Project/Test Set/Crappy-Test-Dataset/127.E 6932 TW-05-18.jpg')...]




```python
test_dl = learn.dls.test_dl(test_files)
```


```python
test_dl.show_batch(max_n=12)
```


    
![png](\images\portfolio\LicensePlateSuperResolution\output_93_0.png)
    



```python
preds, _ = learn.get_preds(dl=test_dl) 
```

    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (generator) that exists in the learner. Use `self.learn.generator` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (critic) that exists in the learner. Use `self.learn.critic` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (gen_mode) that exists in the learner. Use `self.learn.gen_mode` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    





Save the result of the generated images so we can compare them to our original images.


```python
def save_preds(dls, learn):
  "Save away predictions"
  names = dls.dataset.items
  
  preds,_ = learn.get_preds(dl=test_dl)
  for i,pred in enumerate(preds):
      dec = dls.after_batch.decode((TensorImage(pred[None]),))[0][0]
      arr = dec.numpy().transpose(1,2,0).astype(np.uint8)
      Image.fromarray(arr).save(save_path/names[i].name)
```


```python
save_preds(test_dl, learn)
```

    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (generator) that exists in the learner. Use `self.learn.generator` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (critic) that exists in the learner. Use `self.learn.critic` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    /usr/local/lib/python3.7/dist-packages/fastai/callback/core.py:51: UserWarning: You are shadowing an attribute (gen_mode) that exists in the learner. Use `self.learn.gen_mode` to avoid this
      warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
    





## Result Comparison

Finally, let's see a comparison of the original, blurred, and generated image of our license plate from the Indonesian License Plate Test Set.

![png](/images/portfolio/LicensePlateSuperResolution/snippet_3.png)

# Conclusion

In this project, we created a GAN model with Res-Net encoder to reconstruct blur license plates into perceptually readable images. We use a U-net architecture with a pre-trained Res-Net encoder that utilizes skip connections for our generator, we also use a custom critic from the fastAI library which uses dense block as its basic building block while adding spectral normalization at the end. We then pass both our generator and critic to the GAN model. The model will then train in an adversarial fashion — a min-max game that is repeated several times where the generator generates fake samples of images, to fool the discriminator and the discriminator attempts to differentiate the real (ground-truth) and fake images. For training, we trained the model using CCPD (The Chinese City Parking Dataset) and tested the model on the Indonesian License Plate dataset. The results show that our model is able to generate higher resolution and readable license plates from the broken lower resolution input. For future works, implementing other state-of-the-art models might improve the perceptual performance of the generated images
