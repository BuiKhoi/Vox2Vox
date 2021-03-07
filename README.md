# Vox2Vox 3D-GAN for Brain Tumour Segmentation
The original paper for this project can be found here: [Vox2Vox: 3D-GAN for Brain Tumour Segmentation.](https://arxiv.org/abs/2003.13653)

The data used are the ones given for the Brain Tumour Segmentation (BraTS) Challenge 2020 ([link](https://www.med.upenn.edu/cbica/brats2020/data.html)).
* Note: The dataset registration within the owner, you can sign up for that and you may have the dataset within a week. 

# Getting Started
## Installation
* You can clone my repo:
```
git clone https://github.com/BuiKhoi/Vox2Vox
cd Vox2Vox
```
Or the original repo (which contains some bugs and missing files):
```
git clone https://github.com/mdciri/Vox2Vox
cd Vox2Vox
```

* Then install the required requirements (python 3.8.5 is highly recommended):
```
pip install -r requirements.txt
```

## Vox2Vox training:
* You can change the configuration, including data folder and stuff at [config folder](./config/)

* Set `training = True` in [base config file](./config/base_config.py)

* When ready, trigger training process with:
```
python main.py
```

## Vox2Vox testing:
* Set `training = False` in [base config file](./config/base_config.py)

* When ready, trigger training process with:
```
python main.py
```

# Deeper Configuration

* I used weighted for each class [here](./class_weights.py), if you want to chage class weights, please change this file, or replace `class_weights` within [training config](./config/training_config.py)

* Using `Adam Optimizer` is highly recommened with this problem, `SGD Optimizer` is implemented but not properly tested

* If you get `OOM Error`, you can try reducing `batch_size`

* If you want to continue you previous training session, change `continue_training = True` in [training config](./config/training_config.py), we will load the weights from `generator_latest` and `discriminator_latest` files.

* You can specify which model to be tested with `test_weight` in [testing config](./config/testing_config.py), otherwise, we will load `generator_latest` to test