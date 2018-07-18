# Image Colorisation using CNN

Coloring black and white images using a very simple structure of CNN in tensorflow.

## Overview

<<<<<<< HEAD
I used a very basic CNN architecture in an attempt to make a image coloring bot.

In the alpha version, for cheking the core logic of out concepts and for checking the implementation details, I overfitted the model on a single image to check the results, and the results were as following.

<img
	src=/images/alpha/input.png
	align="left"
/>
<img
	src=/images/alpha/results.png
	align="center"
/>
<img
	src=/images/alpha/original.png
	align="right"
/>

### Todo

- [x] Overfit the model on a single image to check implementation - ALPHA verion
- [x] Write scripts to make and use TFRecords for efficient GPU training
- [x] Train the model on a dataset for coloring human portraits - BETA version
=======
### Todo

- [x] Overfit the model on a single image to check implementation
- [x] Write scripts to make and use TFRecords for efficient GPU training
- [ ] Train the model on a dataset for coloring human portraits (Currently working on this)
>>>>>>> 93b3dc2727981a0242eff9fdb8cf027d8ebe3888
- [ ] Use a trained model (Inceptionv3 or ResNet) as a feature extractor for better results
- [ ] Train the model to color any random image

### A simple structure of CNN, as described by the image below was trained from scratch.

Currently, the model is trained only for human portraits.

<img 
<<<<<<< HEAD
	src=/images/graph.png
=======
	src=/utils/graph.png
>>>>>>> 93b3dc2727981a0242eff9fdb8cf027d8ebe3888
	align="left"
/>
