# Image Colorisation using CNN

Coloring black and white images using a very simple structure of CNN in tensorflow.

## Overview

I used a very basic CNN architecture in an attempt to make a image coloring bot.

In the alpha version, for cheking the core logic of out concepts and for checking the implementation details, I overfitted the model on a single image to check the results, and the results were as following.


### Todo

- [x] Overfit the model on a single image to check implementation - ALPHA verion
- [x] Write scripts to make and use TFRecords for efficient GPU training
- [x] Train the model on a dataset for coloring human portraits
- [ ] Train the model to color any random image - BETA version
- [ ] Use a trained model (Inceptionv3 or ResNet) as a feature extractor for better results - FINAL version


### Alpha Version

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


### A simple structure of CNN, as described by the image below was trained from scratch.

Currently, the model is trained only for human portraits.

<img 
	src=/images/graph.png
	align="left"
/>
