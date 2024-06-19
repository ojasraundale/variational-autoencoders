# variational-autoencoders
An implementation of VAEs (variational autoencoders) in Python JAX library. 

## About
This repo is derived from a homework assignment from the course COMPSCI 688: Probabilistic Graphical Models, Spring '24 at the University of Massachusetts, Amherst. 

In brief, the VAE tries to learn a mapping from the distribution of an images $X$ to a latent space $Z$ via an encoder. And the decoder learns a mapping from the latent to the image distribution. A cool thing is that to sample from the model, you just need to sample from the learnt latent space and apply the decoder function. Sampling from the latent is quite easy as it can be done by directly sampling from a multivariate normal distribution. 

[download.png](/download.png) shows 10 generated images from the learned model. 

[image.png](/image.png) plots the latent distribution of the training data. Note that the model never looks at the truth label and still is able to segment the images. 

## Run the notebook
The notebook the run VAE can be found in [vae.ipynb](/src/vae.ipynb)

