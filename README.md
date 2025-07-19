# CE-GNN: Community-Enhanced Graph Neural Network for Elastic Property Prediction of Concrete

This repository provides the code and data for the manuscript:

**"Graph-Based Modeling of Elastic Behavior in Microscale Multi-Phase Concrete using Community-Enhanced GNN"**

## Overview

We propose a CE-GNN framework to predict bulk and shear moduli of 4-phase concrete microstructures using graph neural networks enhanced by community detection. The model transforms 3D voxel-based microstructures into graphs via Mapper clustering and applies a community-aware message passing mechanism.
<img width="1020" height="727" alt="image" src="https://github.com/user-attachments/assets/39ef3ffd-730b-4771-b002-59016d71e773" />

## Features

- Graph construction from 3D X-ray CT images
- Community-aware GNN message passing
- FEM-based ground truth generation
- Supports prediction of K (bulk modulus), μ (shear modulus), and derived properties (E, ν, λ)
- Significantly faster than FEM while maintaining high accuracy


## Code usage
To use this code, follow these steps:

1. Clone the Repository:

```bash 
git clone https://github.com/GMZ774760114/Community-Enhanced-GNN-for-Elastic-Property-Prediction-of-Multi-Phase-Concrete
```



2. Install Dependencies:

```bash 
conda env create -f environment.yml
conda activate CE-GNN_MechanicalProperties_Prediction
```

3. Convert voxel data to graph data:

```bash
python mapper_DFS.py # Take 10 raw voxel 3D data samples as an example
```

4. Add community structure to the graph

```bash
python Community_structure.py # Take the graph generated from step 3 as input
```

5. Train & Evaluate the Model:

```bash
python train_predict.py # Two types of graph data included (with and without community) 
```

