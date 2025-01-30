# GeometricsFinalProjectAviBen

## Overview  
This repository contains the implementation of the final project for the course *Geometry of Information*. The selected paper for the project is **"Learning Parametric Dictionaries for Signals on Graphs"** by Dorina Thanou, David I. Shuman, and Pascal Frossard. The implementation applies the methods described in the paper to synthetic data, traffic congestion data, and brain imaging data.  

## Repository Structure  
- **`data_sets/`**: Contains the processed datasets used in the experiments. The raw data for the traffic and brain datasets were preprocessed using two Python scripts, each specific to the dataset. These preprocessing scripts are also included in this directory.  
- **`results/`**: Contains visualizations of the results obtained from the experiments, including representation errors, learned kernels, and so on.  
- **`optimizers/`**: Contains the optimization solvers required for dictionary learning, which are implemented in MEX.  
- **`synthetic_graph_reconstruction.m`**: Implements the synthetic graph experiment described in the selected paper.  
- **`Traffic_data.m`**: Applies the dictionary learning algorithm to the traffic congestion dataset.  
- **`brain_data.m`**: Applies the dictionary learning algorithm to the brain imaging dataset.  
- **Remaining MATLAB files**: These include helper functions for graph construction, Laplacian computation, and dictionary learning, as well as edited versions of the source code provided with the paper.  

## Installation and Setup  
1. **MATLAB Environment**:  
   Ensure MATLAB is installed with the required toolboxes, including:  
   - Optimization Toolbox  
   - Signal Processing Toolbox  
   - Statistics and Machine Learning Toolbox  

2. **MEX Files**:  
   The dictionary learning algorithm might require the installation of MEX files for optimization solvers. If necessary, consult MATLAB documentation for guidance.  

3. **Adjust File Paths**:  
   The scripts include an `addpath` command at the beginning. You may need to edit this line to match the structure of your local directory before running the code.  
