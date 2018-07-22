# Image Colorisation using CNN

Coloring black and white images using a very simple structure of CNN in tensorflow.

## Overview

I used a very basic CNN architecture in an attempt to make a image coloring bot.

In the alpha version, for checking the core logic of out concepts and for checking the implementation details, I overfitted the model on a single image to check the results, and the results were as following.


### Todo

- [x] Overfit the model on a single image to check implementation - ALPHA verion
- [x] Write scripts to make and use TFRecords for efficient GPU training
- [x] Train the model on a dataset for coloring human portraits
- [ ] Train the model to color any random image - BETA version
- [ ] Use a trained model (Inceptionv3 or ResNet) as a feature extractor for better results - FINAL version


### Alpha Version

At this point of time, the model was overfitted on a single image for 10,000 epochs.
Because of overfitting on a single image, the model gave results with colors close to the original image.

<img
	src=/images/alpha/input.png
	align="center"
/>
<img
	src=/images/alpha/results.png
	align="center"
/>
<img
	src=/images/alpha/original.png
	align="center"
/>


### Beta Version


#### After 20 epochs

At this point of time, the model was trained for 20 epochs with dataset having approximately 900 images of human portraits only.
It is coloring most of the images as brown as that is the most prominent color in human portraits because of human skin. Now I will add a few images with vivid colors in them and reduce the images of human portraits.


<img
	src=/images/beta/input0.png
	align="center"
/>
<img
	src=/images/beta/results0.png
	align="center"
/>
<img
	src=/images/beta/original0.png
	align="center"
/>



#### After 220 epochs


As previously this model was trained only for portraits, it saw a majority of pictures with different textures of brown in the form of skin of people, so it is coloring most of the images brown.
I will run a few more epochs by adding more images with vivid colors in them and that will be the beta version of image coloring model.


<img
	src=/images/beta/input220.png
	align="center"
/>
<img
	src=/images/beta/results220.png
	align="center"
/>
<img
	src=/images/beta/original220.png
	align="center"
/>



<img
	src=/images/beta/input220_1.png
	align="center"
/>
<img
	src=/images/beta/results220_1.png
	align="center"
/>
<img
	src=/images/beta/original220_1.png
	align="center"
/>



<img
	src=/images/beta/input220_2.png
	align="center"
/>
<img
	src=/images/beta/results220_2.png
	align="center"
/>
<img
	src=/images/beta/original220_2.png
	align="center"
/>


### A simple structure of CNN, as described by the image below was trained from scratch.

Currently, the model is trained only for human portraits.

<img 
	src=/images/graph.png
	align="left"
/>
