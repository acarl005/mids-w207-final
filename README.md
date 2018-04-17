# Kaggle Plant Seedlings Classification

My model for competing in [this Kaggle competition](https://www.kaggle.com/c/plant-seedlings-classification).
This is for the MIDS w207 (Applied Machine Learning) final project. 
My final ranking was 528/836.

See [the Jupyter Notebook](./main.ipynb).


## Setup

### To Run Locally
1. Download the necessary Python packages.
1. Download the data from [Kaggle](https://www.kaggle.com/c/plant-seedlings-classification/data), unzip it, and place in `./data/train/raw` and `./data/test/raw`. It should look something like `./data/train/raw/<class-name>/*.png` and `./data/test/raw/*.png`. The training data is organized into subdirectories, named after their respective classes. The test data should have no subdirectories, as it is unlabelled.
1. Open the Jupyter Notebook `main.ipynb`.

### To Run in AWS

1. Do the first 2 steps above.
1. Set up from [this repo](https://github.com/acarl005/aws-terraform-deep-learning).

I trained this on a machine equipped with an Nvidia TitanX GPU, an 8-core Intel i7 CPU, and 64GB of RAM. This notebook won't run on most personal computers.
