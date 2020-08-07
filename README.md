# Super-resolution Autoencoder Experiments
Experiments using sequentially stacked convolutional autoencoders. Not based off of any particular paper.
### Prerequisites

- The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
- A python notebook environment
- Python 3.7+ 
	- TensorFlow 2.0 or greater
	- Pandas
	- Numpy
	- OpenCV3
	- Matplotlib

## Data Processing
1. Data cropped and resized to standard 128x128
2. 64x64 and 32x32 variants created with linear interpolation
3. Variants rescaled back up to 128x128 with linear interpolation
4. Images normalized between (-1, 1) before training, and un-normalized to (0, 255) (int) after inference.

## Model Description
The model consists of two identical convolutional autoencoders, simply stacked back to back. The only difference is that the first autoencoder (AE1) takes the rescaled 32x32 images as input and computes MSE loss against the rescaled 64x64 images, where the second autoencoder (AE2) takes the rescaled 64x64 images as input and computes MSE loss against the 128x128 images.

## Training
- Adam optimizer, LR= 5e-4, beta1 and beta2 default
- 50 epochs
- 4000 training image pairs, 1000 testing image pairs (32, 64, 128 dim each)
- MSE loss

Here is a simple diagram explaining the architecture and where losses are calculated:

![Arch](https://github.com/kevinwoodward/superres-ae/blob/master/imgs/arch.png?raw=true)

## Samples
![Samples 1](https://github.com/kevinwoodward/superres-ae/blob/master/testingsamples/samples1.png?raw=true)

![Samples 2](https://github.com/kevinwoodward/superres-ae/blob/master/testingsamples/samples2.png?raw=true)

## Improvements
- Use a better-suited loss such as [Perceptual Loss](https://arxiv.org/abs/1603.08155)
- Scale to 256x256
- Include more training/testing data