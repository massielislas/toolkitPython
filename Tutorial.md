## Python Toolkit Tutorial

In this short tutorial, we will guide you through setting up the Python Machine Learning toolkit for 478.

### Install

### Your First Script

First, import the required modules.
```
from toolkit import baseline_learner, manager, arff, supervised_learner
import numpy as np
```

# Read in an arff file
```
arff_path = r"./test/datasets/creditapproval.arff"
credit_approval = arff.Arff(arff=arff_path)
```

'credit_approval' is an Arff object. The Arff object is mostly a wrapper around a 2D numpy array. In the case above, this array can be accessed directly as credit_approval.data. The Arff object can also be sliced like traditional numpy arrays. E.g. the first row of data as a numpy array would be:

```
lables = credit_approval[0,:]
```

The Arff object also contains all the information needed to recreate the Arff file. Specifically, it stores attribute names, the number of label columns, whether each variable is nominal/continuous, and the list of possible values for nominal variables. Note that:

# By default, it encodes nominal variables as integers. 
# The toolkit presently supports 1 label, which is assumed to be the rightmost column. There is some partial support for multiple label columns, which are assumed to be the n rightmost columns.

Because slicing the Arff file loses some of the metadata, users can slice ugh...


# Get 1st row of features as an ARFF
features = credit_approval.get_features(slice(0,1))

# Print as arff
print(features)

# Print Numpy array
print(features.data)

# Get all labels as numpy array using slicing


# Manual Training/Test
# Session can take either instantiated or uninstantiated learner

my_learner = baseline_learner.BaselineLearner
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner)
train_features, train_labels, test_features, test_labels = session.training_test_split(.7)  # 70% training
session.train(train_features, train_labels)
session.test(test_features, test_labels)
print(session.training_accuracy)

# Pass on hyperparameters to learner
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner, data=credit_approval, example_hyperparameter=.5)
print(session.learner.data_shape, (690, 16))
print(session.learner.example_hyperparameter, .5)

# Automatic
session2 = manager.ToolkitSession(arff=credit_approval, learner=my_learner, eval_method="random", eval_parameter=.7)

# Cross validate
session3 = manager.ToolkitSession(arff=credit_approval, learner=my_learner)
session3.cross_validate(folds=10, reps=3)
print(session3.test_accuracy)




### Algorithm

<img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/docs/munit_assumption.jpg" width="800" title="Assumption"> 

MUNIT is based on the partially-shared latent space assumption as illustrated in (a) of the above image. Basically, it assumes that latent representation of an image can be decomposed into two parts where one represents content of the image that is shared across domains, while the other represents style of the image that is not-shared across domains. To realize this assumption, MUNIT uses 3 networks for each domain, which are 

1. content encoder (for extracting a domain-shared latent code, content code)
2. style encoder (for extracting a domain-specific latent code, style code)
3. decoder (for generating an image using a content code and a style code)

In the test time as illustrated in (b) of the above image, when we want to translate an input image in the 1st domain (source domain) to a corresponding image in the 2nd domain (target domain). MUNIT first uses the content encoder in the source domain to extract a content codes, combines it with a randomly sampled style code from the target domain, and feed them to the decoder in the target domain to generate the translation. By sampling different style codes, MUNIT generates different translations. Since the style space is a continuous space, MUNIT essentially maps an input image in the source domain to a distribution of images in the target domain.  

### Requirments


- Hardware: PC with NVIDIA Titan GPU. For large resolution images, you need NVIDIA Tesla P100 or V100 GPUs, which have 16GB+ GPU memory. 
- Software: *Ubuntu 16.04*, *CUDA 9.1*, *Anaconda3*, *pytorch 0.4.1*
- System package
  - `sudo apt-get install -y axel imagemagick` (Only used for demo)  
- Python package
  - `conda install pytorch=0.4.1 torchvision cuda91 -y -c pytorch`
  - `conda install -y -c anaconda pip`
  - `conda install -y -c anaconda pyyaml`
  - `pip install tensorboard tensorboardX`

### Docker Image

We also provide a [Dockerfile](Dockerfile) for building an environment for running the MUNIT code.

  1. Install docker-ce. Follow the instruction in the [Docker page](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1)
  2. Install nvidia-docker. Follow the instruction in the [NVIDIA-DOCKER README page](https://github.com/NVIDIA/nvidia-docker).
  3. Build the docker image `docker build -t your-docker-image:v1.0 .`
  4. Run an interactive session `docker run -v YOUR_PATH:YOUR_PATH --runtime=nvidia -i -t your-docker-image:v1.0 /bin/bash`
  5. `cd YOUR_PATH`
  6. Follow the rest of the tutorial.

### Training

We provide several training scripts as usage examples. They are located under `scripts` folder. 
- `bash scripts/demo_train_edges2handbags.sh` to train a model for multimodal sketches of handbags to images of handbags translation.
- `bash scripts/demo_train_edges2shoes.sh` to train a model for multimodal sketches of shoes to images of shoes translation.
- `bash scripts/demo_train_summer2winter_yosemite256.sh` to train a model for multimodal Yosemite summer 256x256 images to Yosemite winter 256x256 image translation.

If you break down the command lines in the scripts, you will find that to train a multimodal unsupervised image-to-image translation model you have to do

1. Download the dataset you want to use. 

3. Setup the yaml file. Check out `configs/demo_edges2handbags_folder.yaml` for folder-based dataset organization. Change the `data_root` field to the path of your downloaded dataset. For list-based dataset organization, check out `configs/demo_edges2handbags_list.yaml`

3. Start training
    ```
    python train.py --config configs/edges2handbags_folder.yaml
    ```
    
4. Intermediate image outputs and model binary files are stored in `outputs/edges2handbags_folder`

### Testing 

First, download our pretrained models for the edges2shoes task and put them in `models` folder.

### Pretrained models 

|  Dataset    | Model Link     |
|-------------|----------------|
| edges2shoes |   [model](https://drive.google.com/drive/folders/10IEa7gibOWmQQuJUIUOkh-CV4cm6k8__?usp=sharing) | 
| edges2handbags |   coming soon |
| summer2winter_yosemite256 |   coming soon |


#### Multimodal Translation

Run the following command to translate edges to shoes

    python test.py --config configs/edges2shoes_folder.yaml --input inputs/edges2shoes_edge.jpg --output_folder results/edges2shoes --checkpoint models/edges2shoes.pt --a2b 1
    
The results are stored in `results/edges2shoes` folder. By default, it produces 10 random translation outputs.

| Input | Translation 1 | Translation 2 | Translation 3 | Translation 4 | Translation 5 |
|-------|---------------|---------------|---------------|---------------|---------------|
| <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/inputs/edges2shoes_edge.jpg" width="128" title="Input"> | <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/results/edges2shoes/output001.jpg" width="128" title="output001"> | <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/results/edges2shoes/output002.jpg" width="128" title="output002"> | <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/results/edges2shoes/output003.jpg" width="128" title="output003"> | <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/results/edges2shoes/output004.jpg" width="128" title="output004"> | <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/results/edges2shoes/output005.jpg" width="128" title="output005"> |


#### Example-guided Translation

The above command outputs diverse shoes from an edge input. In addition, it is possible to control the style of output using an example shoe image.
    
    python test.py --config configs/edges2shoes_folder.yaml --input inputs/edges2shoes_edge.jpg --output_folder results --checkpoint models/edges2shoes.pt --a2b 1 --style inputs/edges2shoes_shoe.jpg
 
| Input Photo | Style Photo | Output Photo |
|-------|---------------|---------------|
| <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/inputs/edges2shoes_edge.jpg" width="128" title="Input"> | <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/inputs/edges2shoes_shoe.jpg" width="128" title="Style"> | <img src="https://raw.githubusercontent.com/NVlabs/MUNIT/master/results/output000.jpg" width="128" title="Output"> |   
 
### Yosemite Summer2Winter HD dataset

Coming soon.


