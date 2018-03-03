# Kaggle Plant Seedlings Classification

Our model for competing in [this Kaggle competition](https://www.kaggle.com/c/plant-seedlings-classification).
This is for the MIDS w207 (Applied Machine Learning) final project. 

See [the Jupyter Notebook](./main.ipynb).


## Setup

### To Run Locally
1. Download the necessary Python packages.
1. Download the data from [Kaggle](https://www.kaggle.com/c/plant-seedlings-classification/data), unzip it, and place in `./data/train/raw` and `./data/test/raw`. It should look something like `./data/train/raw/<class-name>/*.png` and `./data/test/raw/*.png`. The training data is organized into subdirectories, named after their respective classes. The test data should have no subdirectories, as it is unlabelled.
1. Open the Jupyter Notebook `main.ipynb`.

### To Run in AWS

**WARNING: Never, ever copy any AWS credentials into source-controlled repositories. Someone will steal then and mine a bunch of Bitcoin on your dime!**

1. Do the first 2 steps above.
1. Download [Terraform](https://www.terraform.io/downloads.html).
1. Obtain an AWS key and secret and place in `~/.aws/credentials` under the `[terraform]` profile.
1. Create an SSH key pair with `ssh-keygen` and add the private key with `ssh-add`.
1. Go into the `terraform` directory and run `terraform init` and `terraform apply`. You will be prompted to provide the location of your newly generated public SSH key. Assuming you have the correct permissions and such, you should see some output like the public IP addr and DNS name for the newly created instance.
1. Get the address and try logging in via SSH `ssh ubuntu@your-host-name.compute1.amazonaws.com`
1. Mount the EBS Volume with [these instructions](https://devopscube.com/mount-ebs-volume-ec2-instance/). Keep in mind the device location in `/dev` will probably be different.
1. Copy the files to the instance.
1. Activate the Conda environment you want. For example, if you want TensorFlow on Python 3.6: `. ~/anaconda3/bin/activate tensorflow_p36`
1. Run Jupyter `jupyter notebook --no-browser`. Get the token.
1. To access the notebook you can SSH tunnel: `ssh -NL 8157:localhost:8888 ubuntu@your-host-name.compute-1.amazonaws.com`
1. Visit your browser at `http://localhost:8157/?token=<your-token-here>`.

