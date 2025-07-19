# CE-GNN: Community-Enhanced Graph Neural Network for Elastic Property Prediction of Concrete

This repository provides the code and data for the manuscript:

**"Graph-Based Modeling of Elastic Behavior in Microscale Multi-Phase Concrete using Community-Enhanced GNN"**

## Overview

We propose a CE-GNN framework to predict bulk and shear moduli of 4-phase concrete microstructures using graph neural networks enhanced by community detection. The model transforms 3D voxel-based microstructures into graphs via Mapper clustering and applies a community-aware message passing mechanism.

## Features

- Graph construction from 3D X-ray CT images
- Community-aware GNN message passing
- FEM-based ground truth generation
- Supports prediction of K (bulk modulus), μ (shear modulus), and derived properties (E, ν, λ)
- Significantly faster than FEM while maintaining high accuracy

## Repository Structure

