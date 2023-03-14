# Deep Convolutional Generative Adversarial Networks (DCGAN)

This repository contains the code for training and generating images using Deep Convolutional Generative Adversarial Networks (DCGAN). DCGAN is a popular generative model that uses deep convolutional neural networks (CNNs) to generate high-quality images.

DCGAN was introduced in the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Radford et al. (2016). The paper showed that DCGAN can learn to generate realistic images that resemble natural images from datasets such as CIFAR-10 and ImageNet.

In this repository, you can find two Python scripts:

- `dcgan_single.py`: This script trains a DCGAN model on a specified image dataset.
- `generator.py`: This script generates new images using a trained DCGAN model.

The scripts were tested on an AWS g4dn.2xlarge EC2 instance using the `amazon/Deep Learning AMI GPU TensorFlow 2.11.0 (Ubuntu 20.04) 20221220` image.

For more information about DCGAN, please refer to the original paper:

> Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. arXiv preprint arXiv:1511.06434.

I based my code on both these links:

- [Deep Convolutional GAN: How to use a DCGAN to generate images in Python](https://towardsdatascience.com/deep-convolutional-gan-how-to-use-a-dcgan-to-generate-images-in-python-b08afd4d124e)
- [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)

## Train

To train the DCGAN, run the `dcgan_single.py` script with the following required arguments:

- `dataset`: Path to the dataset
- `epochs`: Number of epochs to train
- `batch`: Batch size
- `img-dim`: Input image width and height
- `output`: Model output path

The following optional arguments can also be used:

- `eval`: Evaluate loss every X epochs (default: 100)
- `save`: Save models every X epochs (default: 1000)
- `lat-dim`: Latent dimension scale (default: 100)
- `img-channels`: Input image channels (1 for grayscale) (default: 3)

Example usage:

"""python dcgan_single.py -dataset data/training_images/ -epochs 5000 -batch 64 -img-dim 128 -output output/ -eval 200 -save 1000 -lat-dim 64 -img-channels 3"""


## Generating Images

To generate synthetic images using the trained model, run the `generator.py` script. The script takes the following arguments:

- `generator`: The path to the trained generator model.
- `category`: The category of images to generate.
- `count`: The number of images to generate.
- `output`: The directory where the generated images will be saved.

For example, to generate 50 images of the "dead" category using a trained generator model saved in the `output` directory, run:

"""python generator.py -generator output/generator_20000.h5 -category dead -count 50 -output output/generated_images"""
