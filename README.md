 # Data Grid Replica Placement Optimizer
 
A comprehensive implementation of dynamic programming based replica placement algorithm for data grids, based on the research paper "A Dynamic Programming Based Replica Placement Algorithm in Data Grid". 
The link to the original research paper is given here : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10168787

This project implements and compares two algorithms for optimal replica placement in tree-based data grid structures:

- Dynamic Programming Algorithm - An optimal algorithm that minimizes total communication cost while respecting workload constraints
- Proportional Algorithm - A heuristic approach for comparison

## Key Features

* Interactive Tree Builder: Create custom data grid trees with visual validation
* Algorithm Comparison: Compare DP vs Proportional algorithms across multiple parameters
* Performance Analysis: Analyze algorithm behavior with varying replica counts and workload constraints
* Comprehensive Visualization: Interactive plots, heatmaps, and performance matrices
* Real-time Results: Instant feedback with detailed metrics and statistics

## To run the application , follow these steps 

 ### Prerequisites
      1. Python 3.8+
      2. pip (Python package manager)

### Step 1 : Clone the repository
```bash
git clone https://github.com/your-username/dp-replica-placement.git
cd dp-replica-placement
```
### Step 2 : Set Up Environment
```bash
python -m venv gridreplica_env
source gridreplica_env/bin/activate
```
### Step 3 : Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4 : 
Option 1 : To run the streamlit app
```bash
streamlit run streamlit_app.py
```
Option 2 : To run in Command Line Interface (CLI)
```bash
python main.py --replicas 4 --workload 60 --compare --visualize
```



