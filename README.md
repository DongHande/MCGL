# Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning

This repository is the official implementation of [Data Augmentation View on Graph Convolutional Network and the Proposal of Monte Carlo Graph Learning](url_arxiv).

## Overview

In this project, we provide implementations of visual analysis and three models -- GCN, GCN* and MCGL-UM. The repository is organised as follows:

- `data/` contains the necessary dataset files for CORA, PubMed, CiteSeer and MS Academic.
- `plot/` contains the visual analysis on synthetic clean and noisy graphs.
- `train_MC_base.py` is the implementations of MCGL-UM.
- `train_GCN.py` is the implementations of GCN and GCN*.
- `layers.py` contains the definition of one-layer linear transformation, one-layer GCO.
- `models.py` contains the GCN and MLP.
- `ps.py` contains the command-line arguments definition with default values. By specifying arguments in this file, you can change the dataset and adjust hyper-parameters.
- `utils.py` contains preprocessing utilities of data loading, data split, and noise reducing, 


## Requirements

The project runs under Python 3.8.3 with several required dependencies:

* numpy==1.18.4
* scipy==1.4.1
* matplotlib==3.2.1
* networkx==2.4
* torchvision==0.6.0
* torch==1.5.0

In addition, CUDA 10.2 is used in this project.

## Visual analysis

This paper has some visual analysis. Run commands below to get these figures. 

```visual
cd plot
python comparison_clean.py
python comparison_noisy.py
python deep_GCO.py
```

## Models

In the paper, we used three different models -- GCN, GCN* and MCGL-UM. To implement them, run respective commands below:

```train models
python train_GCN.py --baseline 1

python train_GCN.py --baseline 2

python train_MC_base.py
```

By specifying arguments in 'ps.py', you can change the dataset and adjust hyper-parameters.
The optimal hyper-parameters we have drawn are shown below: hidden units/weight decay/learning rate/dropout rate/(batch size for MCGL-UM)


Dataset | GCN | GCN* | MCGL-UM
:-: | :-: | :-: | :-:
CORA | 32/0.0005/0.005/0.7 | 32/0.0005/0.01/0.7 | 32/0.001/0.005/0.5/50
CiteSeer | 64/0.001/0.05/0.6 | 64/0.001/0.05/0.4 | 64/0.001/0.005/0.3/200
PubMed | 32/0.0005/0.05/0.3 | 32/0.0005/0.005/0.5 | 32/0.001/0.005/0.5/50
MS Academic | 128/0.0005/0.01/0.6 | 128/0.0005/0.005/0.7 | 128/0.0001/0.005/0.5/200

By specifying arguments ("depth", "trdep", "tsdep", and "noise_rate"), you can implement different depth of GCN* and MCGL-UM, and different reduced noise rate of all three models, for example:

```train depth and nr
python train_GCN.py --baseline 2 --depth 10

python train_MC_base.py --trdep 10 --tsdep 10

python train_GCN.py --baseline 1 --noise_rate 0.1
```

## Datasets

In this paper, we use three citation datasets and a co-authorship dataset, 
which can be downloaded in https://github.com/tkipf/gcn/tree/master/gcn/data and https://github.com/klicperajo/ppnp/tree/master/ppnp/data. 
CORA, CiteSeer and PubMed are citation graphs, where a node represents a paper, and an edge between two nodes represents that the two papers have a citation relationship.
MS Academic is co-author graph, an edge in the graph represents the co-authorship between two papers.
