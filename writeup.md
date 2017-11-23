
## Follow Me: Deep Learning Project ##

This project trains a deep neural network to identify and track a target in simulation.

### Neural Network Architecture

In this project, we are using a Fully Convolutional Neural Network (FCN) to help us in image segmentation and object identification.

A typical FCN is comprised of multiple encoder blocks, followed by 1x1 convolutional layers, then followed by decoder blocks. In our case, the network layout have 3 encoder blocks with a 1x1 convolutional layer and 3 encoder blocks. The encoder blocks extract features of the image, the decoder blocks upscale the output back to the size of the original image.

<img src="./docs/misc/fcn1.png" width="500">

### Setting the Network Parameters

#### Learning Rate
The learning rate is the amount of correction which the network applies when modifing the weights. For our case, I tried from 0.01 to 0.0001, and finally chose 0.001 as the learning rate.

#### Batch Size

Batch size defines number of samples that going to be propagated through the network. For our case, I set the batch size to 40.

#### num_epochs
An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data. For our case, I set the num_epochs to 60.


#### steps_per_epoch
The steps_per_epoch is number of batches of training images that go through the network in 1 epoch. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.  For our case, I set the steps_per_epoch to 200.

#### validation_steps
The validation_steps is number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. For our case, I set the validation_steps to 50.


### Neural Network Constructions

#### Fully Connected Layers

A fully connected layer multiplies the input by a weight matrix and then adds a bias vector. It is useful for image recognize, but the spatial information are lost, so, the spatial information of the images are missing.

<img src="./docs/misc/fnn.png" width="500">


#### 1x1 Convolutional Layers

1x1 Convolutional Layer is a neural network layer that is filtered by a 1x1 filter with the step size of one. The advantages of 1x1 Convolutional Layer is that it is a cheap way to make models deeper and have more parameters, without completely changing their structures.

Meanwhile, by replacement of fully-connected layers with convolutional layers,  spatial information is preserved.

<img src="./docs/misc/Selection_001.png" width="500">

#### Separable Convolutions

Separable convolutions, also known as depthwise separable convolutions, comprise of a convolution performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer. It can reduce the number of parameters needed, thus increasing efficiency for the encoder network.

For instance, if an input shape of 32x32x3, the desired number of 9 output channels and filters (kernels) of shape 3x3x3. regular convolutions would result in a total of 243 parameters, but with a separable convolutions, a total of 54 (27 + 27) parameters are needed.


#### Bilinear Upsampling
Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value.

<img src="./docs/misc/bilinear.png" width="500">

### Results and Discussion

The trained network are tested in simulation, it works well in the simulation. However, the target person tracked in the simulation has different colors comapred with other person, so it will be easier compared to real life.

The network should be able to be used for recognizing other objects, but the parameters maybe not the same. There should be a large batch of images, then using the images and the model to train and tune the parameters, finally the trained model can be used for recognizing other objects.

I uploaded a video to youtube. The link is as follows.

[![https://youtu.be/gE6jEaOTTOU](https://img.youtube.com/vi/gE6jEaOTTOU/0.jpg)](https://www.youtube.com/watch?v=gE6jEaOTTOU)