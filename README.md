[![Triplet loss](https://github.com/karanrampal/triplet-loss/actions/workflows/tripletloss.yaml/badge.svg)](https://github.com/karanrampal/triplet-loss/actions/workflows/tripletloss.yaml)

# Triplet Loss

This is a pytorch implementation of the triplet loss. It is based on the blog post by [Olivier Moindrot](https://omoindrot.github.io/triplet-loss) and the [Facenet paper](https://arxiv.org/abs/1503.03832) by Florian Schroff, Dmitry Kalenichenko and James Philbin.

## Directory structure
Structure of the project
```
.github/
    workflow/
        tripletloss.yaml
experiments/
    base_model/
        params.json
    param_search/
        params.json
model/
    __init__.py
    data_loader.py
    net.py
    triplet_loss.py
tests/
    __init__.py
    test_triplet_loss.py
.gitignore
evaluate.py
LICENSE
Makefile
README.md
requirements.txt
search_hyperparams.py
synthesize_results.py
tox.ini
train.py
utils.py
visualization.py
```

## Usage
The simplest way to use this repository as a template for a project is to clone it and then delete the `.git` directory. Then git can be re-initialized,
```
git clone <url> <newprojname>
cd <newprojname>
```
To start training we can do the following,
```
python train.py --data_dir=<wherever your dataset is>
```
To visualize the trained model, we can do the following,
```
python visualization.py
tensorboard --logdir=experiments/
```

## Requirements
I used Anaconda with python3,

```
conda create -n <yourenvname> python=<3.x>
conda activate <yourenvname>
conda install -n <yourenvname> --file requirements.txt
```
